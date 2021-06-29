import numpy as np
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Implementation based the PyTorch DQN tutorial
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition namedtuple"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x, device='cpu'):
        x = x.to('cpu')
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class AgentDDQN(object):

    def __init__(self, stateSize=40, memory_size=1000, batch_size=128, gamma=0.999,
                 eps_start=0.9, eps_end=0.05, eps_decay=200, target_update=10, device='cpu'):
        self.h = stateSize
        self.w = stateSize
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.device = device

        self.policy_net = DQN(self.h, self.w, 9).to(device)
        self.target_net = DQN(self.h, self.w, 9).to(device)
        self.updateTarget(0)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.000025)
        # self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(memory_size)

        self.steps_done = 0

    def selectAction(self, state):
        self.steps_done += 1
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(9)]], device=self.device, dtype=torch.long)

    def optimizeModel(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def updateTarget(self, eps_idx):
        if eps_idx % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

    def loadParameters(self, checkPointFilePath):
        checkpoint = torch.load(checkPointFilePath)
        self.policy_net.load_state_dict(checkpoint)
        self.target_net.load_state_dict(checkpoint)
