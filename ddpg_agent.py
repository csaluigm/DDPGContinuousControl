import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from config import Config
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

print(torch.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
class Agent():
    def __init__(self, state_size, action_size, seed):
        self.gradient_clipping = True
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.config = Config()
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config.LR_CRITIC, weight_decay=self.config.WEIGHT_DECAY)

        self.noise = OUNoise(action_size, seed)

        self.memory = ReplayBuffer(action_size, self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed, device)
        self.step_count = 0
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1

        if len(self.memory) > self.config.BATCH_SIZE and self.step_count % self.config.UPDATE_EVERY == 0:
            experiences = self.memory.sample()
            self.learn(experiences, self.config.GAMMA)

    def act(self, state, eps, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action = action + self.noise.sample() * eps
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        if self.gradient_clipping:
            # use gradient clipping
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.step_count % 10 == 0:
            self.soft_update(self.critic_local, self.critic_target, self.config.TAU)
            self.soft_update(self.actor_local, self.actor_target, self.config.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
