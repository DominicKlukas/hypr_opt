import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MLPSoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_dim = int(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


class MLPActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_dim = int(np.array(env.single_observation_space.shape).prod())
        out_dim = int(np.prod(env.single_action_space.shape))

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.fc_mean = nn.Linear(256, out_dim)
        self.fc_logstd = nn.Linear(256, out_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = self.shared(x)
        x = F.relu(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
