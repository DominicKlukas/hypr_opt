import numpy as np
import torch
import torch.nn as nn

import escnn.gspaces as gspaces
import escnn.nn as enn


LOG_STD_MAX = 2
LOG_STD_MIN = -5


def _obs_repr_list(r2_act, obs_dim: int):
    if obs_dim == 19:
        return 5 * [r2_act.irrep(1, 1), r2_act.trivial_repr] + 4 * [r2_act.trivial_repr]
    if obs_dim == 6:
        return 2 * [r2_act.irrep(1, 1), r2_act.trivial_repr]
    raise ValueError(f"Unsupported Fetch observation dim: {obs_dim}")


class EquivariantSoftQNetwork(nn.Module):
    def __init__(self, env, dihedral_n: int = 4, reg_rep_n: int = 16):
        super().__init__()
        r2_act = gspaces.flipRot2dOnR2(dihedral_n)
        obs_dim = int(np.array(env.single_observation_space.shape).prod())

        act_repr_list = [r2_act.irrep(1, 1)] + 2 * [r2_act.trivial_repr]
        obs_repr_list = _obs_repr_list(r2_act, obs_dim)
        self.input_type = enn.FieldType(r2_act, obs_repr_list + act_repr_list)

        layer1_type = enn.FieldType(r2_act, reg_rep_n * [r2_act.regular_repr])
        layer2_type = enn.FieldType(r2_act, reg_rep_n * [r2_act.regular_repr])
        output_type = enn.FieldType(r2_act, [r2_act.trivial_repr])

        self.net = nn.Sequential(
            enn.R2Conv(self.input_type, layer1_type, kernel_size=1, stride=1, padding=0, initialize=True),
            enn.ReLU(layer1_type),
            enn.R2Conv(layer1_type, layer2_type, kernel_size=1, stride=1, padding=0, initialize=True),
            enn.ReLU(layer2_type),
            enn.R2Conv(layer2_type, output_type, kernel_size=1, stride=1, padding=0, initialize=True),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        b = x.shape[0]
        x = x.view(b, -1, 1, 1)
        x = self.input_type(x)
        return self.net(x).tensor.view(b, -1)


class EquivariantActor(nn.Module):
    def __init__(self, env, dihedral_n: int = 4, reg_rep_n: int = 16):
        super().__init__()
        r2_act = gspaces.flipRot2dOnR2(dihedral_n)
        obs_dim = int(np.array(env.single_observation_space.shape).prod())

        act_repr_list = [r2_act.irrep(1, 1)] + 2 * [r2_act.trivial_repr]
        obs_repr_list = _obs_repr_list(r2_act, obs_dim)

        self.input_type = enn.FieldType(r2_act, obs_repr_list)
        layer1_type = enn.FieldType(r2_act, reg_rep_n * [r2_act.regular_repr])
        layer2_type = enn.FieldType(r2_act, reg_rep_n * [r2_act.regular_repr])
        output_type = enn.FieldType(r2_act, act_repr_list)

        self.fc1 = enn.R2Conv(self.input_type, layer1_type, kernel_size=1, stride=1, padding=0, initialize=True)
        self.fc1_relu = enn.ReLU(layer1_type)
        self.fc2 = enn.R2Conv(layer1_type, layer2_type, kernel_size=1, stride=1, padding=0, initialize=True)
        self.fc2_relu = enn.ReLU(layer2_type)
        self.fc_mean = enn.R2Conv(layer2_type, output_type, kernel_size=1, stride=1, padding=0, initialize=True)
        self.fc_logstd = enn.R2Conv(layer2_type, output_type, kernel_size=1, stride=1, padding=0, initialize=True)

        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        b = x.shape[0]
        x = x.reshape(b, -1, 1, 1)
        x = self.input_type(x)
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2_relu(self.fc2(x))
        mean = self.fc_mean(x).tensor.view(b, -1)
        log_std = self.fc_logstd(x).tensor.view(b, -1)
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
