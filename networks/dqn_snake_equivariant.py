import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn


class EquivariantSnakeQNetwork(nn.Module):
    """Equivariant Q-network extracted from atari-snake-dqn-d branch.

    Expects stacked RGB history observations shaped (B, 12, 32, 32).
    """

    def __init__(self, num_actions: int):
        super().__init__()
        r2_act = gspaces.flipRot2dOnR2(N=4)
        self.input_type = enn.FieldType(r2_act, 12 * [r2_act.trivial_repr])
        layer1_type = enn.FieldType(r2_act, 8 * [r2_act.regular_repr])
        layer2_type = enn.FieldType(r2_act, 12 * [r2_act.regular_repr])
        layer3_type = enn.FieldType(r2_act, 12 * [r2_act.regular_repr])
        layer4_type = enn.FieldType(r2_act, 32 * [r2_act.regular_repr])

        self.features = enn.SequentialModule(
            enn.R2Conv(self.input_type, layer1_type, kernel_size=7, stride=2, padding=2, bias=False),
            enn.ReLU(layer1_type, inplace=True),
            enn.R2Conv(layer1_type, layer2_type, kernel_size=5, stride=2, padding=1, bias=False),
            enn.ReLU(layer2_type, inplace=True),
            enn.R2Conv(layer2_type, layer3_type, kernel_size=5, stride=1, padding=1, bias=False),
            enn.ReLU(layer3_type, inplace=True),
            enn.R2Conv(layer3_type, layer4_type, kernel_size=7, stride=1, padding=1, bias=False),
            enn.ReLU(layer4_type, inplace=True),
            enn.GroupPooling(layer4_type),
        )
        self.head = nn.Linear(self.features.out_type.size, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 12, 32, 32)
        x = self.features(self.input_type(x / 255.0)).tensor
        x = x.view(x.size(0), -1)
        return self.head(x)
