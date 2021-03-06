import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

print("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:[%s]."%(device))


# Actor Layer Construction
class ActorClass(nn.Module):

    def __init__(self, name='Actor', state_dim=4, action_dim=1, action_bound=1, learning_rate=1e-2, hdims=[64, 32, 16]):

        # class initialize
        super(ActorClass, self).__init__()

        # ActorClass parameter initialize
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.hdims = hdims

        self.layers = []

        # Dense Layer construction
        prev_hdim = self.state_dim
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(nn.ReLU())  # activation function = relu
            prev_hdim = hdim

        # Final Layer (without activation)
        #print(action_dim)
        self.layers.append(nn.Linear(prev_hdim, 1))

        # Concatenate all layers
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.net(x)
        mu = self.action_bound * torch.tanh(x)
        std = F.softplus(x)
        return mu, std


# Critic Layer Construction
class CriticClass(nn.Module):
    def __init__(self, name='Critic', state_dim=4, learning_rate=1e-2, hdims = [64, 32, 16]):

        # class initialize
        super(CriticClass, self).__init__()

        # ActorClass parameter initialize
        self.state_dim = state_dim
        self.hdims = hdims

        self.layers = []

        # Dense Layer construction
        prev_hdim = self.state_dim
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(nn.ReLU())  # activation function = relu
            prev_hdim = hdim

        # Final Layer (without activation)
        self.layers.append(nn.Linear(prev_hdim, 1))

        # Concatenate all layers
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, x):
        val = self.net(x)  # scalar
        return val

# PPO Class + train net
class PPO:
    def __init__(self, state_dim=4, action_dim=2, action_bound=1, learning_rate=1e-2, eps_clip=0.2, K_epoch=3, gamma=0.98, lmbda=0.95, buffer_size=30, minibatch_size=32):

        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.lmbda = lmbda
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.optimization_step = 0

        # model
        self.actor = ActorClass(state_dim=state_dim, action_dim=action_dim, action_bound=action_bound, learning_rate=learning_rate)
        self.critic = CriticClass(state_dim=state_dim, learning_rate=learning_rate)
        self.data = []

    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition

                    s_lst.append(s)  # s:numpy array, shape:[...]
                    a_lst.append([a])  # s??? shape??? ????????? ?????? []??? ??????./ pytorch unsqueeze(?????? ??????.)
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    if done:
                        done_mask = 0
                    else:
                        done_mask = 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)

            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float),\
                         torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                         torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)
        return data

    def put_data(self, transition):
        self.data.append(transition)

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.critic.forward(s_prime) * done_mask
                delta = td_target - self.critic.forward(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    def train_net(self):
        if len(self.data) == self.minibatch_size * self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(self.K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.actor.forward(s)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic.forward(s), td_target)

                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()

                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

                    self.actor.optimizer.step()
                    self.critic.optimizer.step()

                    self.optimization_step += 1
                    #print(self.optimization_step)

def main():

    ####### Hyperparameters #######
    env = gym.make('Pendulum-v0')
    score = 0.0
    print_interval = 20

    eps_clip = 0.2
    K_epoch = 10
    lmbda = 0.95
    gamma = 0.98
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space
    action_bound = 1
    learning_rate = 1e-2
    rollout_len = 5
    buffer_size = 30
    minibatch_size = 32
    ###############################

    agent = PPO(state_dim=state_dim,action_dim=action_dim,action_bound=action_bound, learning_rate=learning_rate,\
                eps_clip=eps_clip,K_epoch=K_epoch,gamma=gamma,lmbda=lmbda, buffer_size=buffer_size,minibatch_size=minibatch_size)

    rollout = []

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(rollout_len):
                mu, std = agent.actor(torch.from_numpy(s).float())
                dist = Normal(mu,std)
                #env.render()
                a = dist.sample()
                log_prob = dist.log_prob(a)
                s_prime, r, done, info = env.step([a.item()])

                rollout.append((s, a, r / 10.0, s_prime, log_prob.item(), done))
                if len(rollout) == rollout_len:
                    agent.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                if done:
                    break

            agent.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score / print_interval, agent.optimization_step))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()