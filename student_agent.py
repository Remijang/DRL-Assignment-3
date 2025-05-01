import numpy as np
from collections import deque
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import cv2
import random

class DualDQN(nn.Module):
    def __init__(self, n_channel, n_action):
        super().__init__()
        self.n_action = n_action
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channel, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.adv = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )

        self.val = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
        x = x.float() / 255.0
        y = self.conv_net(x)
        adv = self.adv(y)
        val = self.val(y)
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        return q_values

class Agent(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved_model_path = 'model_9500'
        self.action_space = gym.spaces.Discrete(12)
        self.online_net = DualDQN(4, 12).to(self.device)
        saved_data = torch.load(saved_model_path, map_location=self.device)
        self.online_net.load_state_dict(saved_data['online_model_state_dict'])
        self.online_net.eval()

        self.skip = 4
        self.frames = deque(maxlen=4)

        self.start = False
        self.step_count = 0
        self.last_action = 0
        self.size = (84, 84)

    def reset(self):
        self.frames.clear()
        self.start = False
        self.step_count = 0
        self.last_action = 0

    def resize(self, obs):
        obs_tensor = torch.from_numpy(obs.copy()).permute(2, 0, 1).to(self.device)
        obs_gray = TF.rgb_to_grayscale(obs_tensor)
        obs_resized = TF.resize(obs_gray, self.size)
        return obs_resized.byte()

    def act(self, obs):
        if not self.start:
            self.reset()
        new_obs = self.resize(obs)
        change = not self.start or (self.step_count % 4 == 0)
        if change:
            if not self.start:
                for _ in range(4):
                    self.frames.append(new_obs.clone())
                self.start = True
            else:
                self.frames.append(new_obs.clone())
        state = torch.cat(list(self.frames), dim=0)
        tensor = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(tensor)
            self.last_action = q.argmax(dim=1).item()
        self.step_count += 1
        return self.last_action
