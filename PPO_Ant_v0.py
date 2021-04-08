import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pybullet_envs
import time


# Configuration, set model parameter
class Config:
    def __init__(self):
        env = gym.make('AntBulletEnv-v0')
        self.type = "ppo"

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.gae_lmbda = 0.99
        self.learning_rate = 0.0003
        self.learn_interval = 100

        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.n_episode = 300
        self.n_epochs = 1000
        self.batch_size = 5000

class Model(nn.Module):
    def __init__(self, input_shape, outputs_shape):
        super(Model, self).__init__()

        # for gpu setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # state shape
        self.input_shape = input_shape
        # action shape
        self.outputs_shape = outputs_shape

        # first layer size
        features_count = 256

        # first dense layer with activation function
        self.features_layers = nn.Sequential(
            nn.Linear(self.input_shape, features_count),
            nn.ReLU()
        )
        self.features_layers.to(self.device)

        # [Actor_policy layer] mu (mean) dense layer with activation function
        self.layers_mu = nn.Sequential(
            nn.Linear(features_count, 64),
            nn.ReLU(),
            nn.Linear(64, self.outputs_shape),
            nn.Tanh()   # -1 <= mu <= 1
        )
        self.layers_mu.to(self.device)

        # [Actor_policy layer] var (variance) dense layer with activation function
        self.layers_var = nn.Sequential(
            nn.Linear(features_count, 64),
            nn.ReLU(),
            nn.Linear(64, self.outputs_shape),
            nn.Softplus()   # var is bigger than 0
        )
        self.layers_var.to(self.device)

        # [Critic_value layer]
        self.layers_critic = nn.Sequential(
            nn.Linear(features_count, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # output must be scalar value
        )
        self.layers_critic.to(self.device)

    def forward_pi(self, state):
        # go to first layer
        features_output = self.features_layers(state)
        # first layer to mu layer
        mu = self.layers_mu(features_output)
        # first layer to var layer
        std = self.layers_var(features_output)
        dist = Normal(mu,std)
        return dist

    def forward_val(self, state):
        # go to first layer
        features_output = self.features_layers(state)
        # first layer to critic layer
        val = self.layers_critic(features_output)
        return val

class Memory:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store_memory(self, state, action, prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def generate_batch_index(self):
        n_states = len(self.states)
        # This is for sampling a part of trajectory. (t_0, t_1, ..., t_{k+1})
        start_idx = np.arange(0, n_states, self.batch_size)
        idxs = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(idxs)  # To mitigate correlation
        batches = np.split(idxs, len(start_idx))
        return batches

    def get_memory(self):
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.values), np.array(self.rewards), np.array(self.dones)

class Agent:
    def __init__(self, config=Config()):
        self.config = config

        # PPO Hyperparameters
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.gae_lmbda = config.gae_lmbda
        self.learning_rate = config.learning_rate
        self.n_episode = config.n_episode
        self.n_epochs = config.n_epochs
        self.batch_size = config.batch_size
        self.learn_interval = config.learn_interval
        self.observation_space = config.observation_space
        self.action_space = config.action_space

        self.model = Model(self.observation_space, self.action_space)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.memory = Memory(self.batch_size)

    def put_data(self, state, action, log_prob, value, reward, done):
        self.memory.store_memory(state, action, log_prob, value, reward, done)

    def get_policy_action(self, state):
        s = torch.tensor([state], dtype=torch.float).to(self.model.device)
        dist = self.model.forward_pi(s)
        val = self.model.forward_val(s)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        log_prob = torch.squeeze(log_prob).detach().to("cpu").numpy()
        action = torch.squeeze(action).detach().to("cpu").numpy()
        val = torch.squeeze(val).detach().to("cpu").numpy()
        return action, log_prob, val

    def calc_advantage(self, reward_arr, values_arr, dones_arr):
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr) - 1):  # Algorithm1: inner loop
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                # \hat{A} = \delta_t + (\gamma \lambda)\delta_{t+1} + \cdots (in paper (11), (12))
                a_t += discount * (reward_arr[k] + self.gamma * values_arr[k+1] * (
                    1-int(dones_arr[k])) - values_arr[k])  # (12)
                discount *= self.gamma * self.gae_lmbda
            advantage[t] = a_t
            return advantage

    def train_net(self):
        for _ in range(self.n_epochs):  # Algorithm 1: outer loop
            state_arr, action_arr, old_log_probs_arr, values_arr, reward_arr, dones_arr = self.memory.get_memory()
            batches_idx = self.memory.generate_batch_index()

            advantage = self.calc_advantage(reward_arr,values_arr,dones_arr)

            advantage = torch.tensor(advantage, dtype=torch.float).to(self.model.device)
            values_arr = torch.tensor(values_arr, dtype=torch.float).to(self.model.device)

            for idxs in batches_idx:
                states = torch.tensor(state_arr[idxs], dtype=torch.float).to(self.model.device)
                old_log_probs = torch.tensor(old_log_probs_arr[idxs]).to(self.model.device)
                actions = torch.tensor(action_arr[idxs]).to(self.model.device)

                dist = self.model.forward_pi(states)
                value = self.model.forward_val(states)
                value = torch.squeeze(value)

                new_log_probs = dist.log_prob(actions)
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()

                surr1 = advantage[idxs] * prob_ratio.mean(dim=1)
                surr2 = torch.clamp(prob_ratio, 1-self.eps_clip, 1+self.eps_clip).mean(dim=1) * advantage[idxs]

                actor_loss = -torch.min(surr1, surr2)
                target_q_value = (advantage[idxs] + values_arr[idxs]).detach()
                critic_loss = F.smooth_l1_loss(value, target_q_value)

                total_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.mean().backward()
                self.optimizer.step()

        self.memory.clear_memory()

def main():

    env = gym.make('AntBulletEnv-v0')       #'Pendulum-v0','AntPyBulletEnv-v0'
    agent = Agent(config=Config())
    best_score = env.reward_range[0]
    score_history = []

    n_steps = 0
    n_episode = 300
    learn_interval = 100

    for i in range(n_episode):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action, prob, val = agent.get_policy_action(observation)
            env.render()
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.put_data(observation, action, prob, val, reward, done)

            if n_steps % learn_interval == 0:
                agent.train_net()

            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print(f'Episode: {i} / Score: {score} / AVG Score (100): {avg_score}')

    env.close()

if __name__ == '__main__':
    main()


