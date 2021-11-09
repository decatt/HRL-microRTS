import torch.nn as nn
import torch
import numpy as np
from mask import CategoricalMasked

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Critic(nn.Module):
    def __init__(self, envs):
        super(Critic, self).__init__()

        self.state = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            # layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(), )
        self.action = nn.Sequential(
            layer_init(nn.Linear(334, 256)),
            nn.ReLU(), )
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x, a):
        res = self.state(x.permute((0, 3, 1, 2))) + self.action(a)  # "bhwc" -> "bchw"
        res = self.critic(res)
        return res


