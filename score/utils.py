import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, num_ensemble, max_size=int(1e6), ber_mean=1.0, state_norm=True, reward_norm=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.num_ensemble = num_ensemble
        self.max_size = max_size
        self.ber_mean = ber_mean
        self.state_norm = state_norm
        self.reward_norm = reward_norm
        self.init_buffer()

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.mask[self.ptr] = torch.bernoulli(self.ber_mean*torch.ones(self.num_ensemble)).numpy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.mask[ind]).to(self.device),
        )


    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1,1)
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        self.size = self.state.shape[0]


    def set_mask(self):
        self.mask = torch.bernoulli(self.ber_mean*torch.ones(self.size, self.num_ensemble)).numpy()


    def normalize_states(self, eps = 1e-3):
        self.state_mean = self.state.mean(0,keepdims=True)
        self.state_std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - self.state_mean)/self.state_std
        self.next_state = (self.next_state - self.state_mean)/self.state_std
        return self.state_mean, self.state_std

    def normalize_state(self, state):
        return (state - self.state_mean)/self.state_std

    def normalize_rewards(self):
        self.max_reward = self.reward.max()
        self.min_reward = self.reward.min()
        self.reward = (self.reward - self.min_reward)/(self.max_reward-self.min_reward)
        

    def normalize_reward(self, reward):
        return (reward - self.min_reward)/(self.max_reward-self.min_reward)

    def normalize_rewards_2(self, eps = 1e-3):
        self.reward_mean = self.reward.mean(0,keepdims=True)
        self.reward_std = self.reward.std(0,keepdims=True) + eps
        self.reward = (self.reward - self.reward_mean)/self.reward_std

    def init_buffer(self):
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        self.mask = np.zeros((self.max_size, self.num_ensemble))
