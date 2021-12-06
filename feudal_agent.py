import torch.nn as nn
import torch
import numpy as np
from mask import CategoricalMasked

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Manager(nn.Module):
    def __init__(self):
        super(Manager, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(12, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 256)),
            nn.ReLU(),
        )
        """
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            # layer_init(nn.Linear(32 * 6 * 6, 256)),
            layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(),
        )"""
        self.actor = layer_init(nn.Linear(256, 24), std=1)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        # return self.network(x.permute((0, 3, 1, 2)))  # "bhwc" -> "bchw"
        return self.network(x)

    def get_goal(self, x, n):
        num_env = n
        g = self.actor(self.forward(x))
        distributes = torch.where(g > 0, g, torch.tensor(0.000001).to(device))
        goals = torch.zeros((num_env, 12)).to(device)
        entropy = torch.zeros((num_env, 12)).to(device)
        log_probs = torch.zeros((num_env, 12)).to(device)
        for i in range(num_env):
            dp = distributes[i].view(2, -1)
            dd = torch.distributions.Normal(dp[0], dp[1])
            goal = dd.sample()
            goals[i] = goal
            log_prob = dd.log_prob(goal)
            log_probs[i] = log_prob
            entropy[i] = log_prob * goal
        return goals, log_probs.sum(1), entropy.sum(1)

    def get_value(self, x):
        return self.critic(self.forward(x))


class ManagerScript:
    def __init__(self, n):
        self.goal = torch.zeros((n, 12)).to(device)

    def get_goal(self, x, n):
        self.goal = torch.zeros((n, 12)).to(device)
        for i in range(n):
            obs = x[i]
            if obs[2] <= 2:
                self.goal[i][2] = 1
            if obs[1] < 1 and obs[2] >= 2:
                self.goal[i][1] = 1
            if obs[5] < 10:
                self.goal[i][5] = 1
            if obs[6] > 0:
                self.goal[i][6] = -1
            if obs[7] > 0:
                self.goal[i][7] = -1
            if obs[8] > 0:
                self.goal[i][8] = -1
            if obs[9] > 0:
                self.goal[i][9] = -1
            if obs[10] > 0:
                self.goal[i][10] = -1
            if obs[11] > 0:
                self.goal[i][11] = -1
        return x + self.goal


class Worker(nn.Module):
    def __init__(self, envs):
        super(Worker, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            # layer_init(nn.Linear(32 * 6 * 6, 256)),
            layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(), )
        self.goal = nn.Sequential(
            layer_init(nn.Linear(12, 256)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x, g):
        return self.network(x.permute((0, 3, 1, 2))) + self.goal(g)  # "bhwc" -> "bchw"

    def get_action(self, x, g, num_envs, action=None, invalid_action_masks=None, envs=None):
        logits = self.actor(self.forward(x, g))
        split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

        if action is None:
            # 1. select source unit based on source unit mask
            source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1))
            multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
            action_components = [multi_categoricals[0].sample()]
            # 2. select action type and parameter section based on the
            #    source-unit mask of action type and parameters
            source_unit_action_mask = torch.Tensor(
                np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(num_envs,
                                                                                                         -1))
            split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
            multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                                       zip(split_logits[1:], split_suam)]
            invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
            action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
            action = torch.stack(action_components)
        else:
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, log_prob.sum(0), entropy.sum(0), invalid_action_masks

    def get_value(self, x, g):
        return self.critic(self.forward(x, g))


"""
class ManagerScript():
    def __init__(self, map_size, num_env):
        self.size = map_size
        self.n = num_env

    def get_goals(self):
        goals = torch.zeros((self.n, 12)).to(device)
        for i in range(self.n):
            goal = torch.Tensor([1, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            goals[i] = goal
        return goals
"""


class ScriptManager(nn.Module):
    def __init__(self):
        super(ScriptManager, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(256, 512), std=1)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))

    def get_goal(self, x, source_unit_mask, action=None):
        logits = self.actor(self.forward(x))
        split_logits = torch.split(logits, [256, 256], dim=1)
        unit_mask = torch.split(source_unit_mask, [256, 256], dim=1)
        if action is None:
            multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=unit_mask[0])]+[CategoricalMasked(logits=split_logits[1], masks=unit_mask[1])]
            action_components = [categorical.sample() for categorical in multi_categoricals]
            action = torch.stack(action_components)
        else:
            split_invalid_action_masks = torch.split(source_unit_mask, [256, 256], dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, log_prob.sum(0), entropy.sum(0), source_unit_mask

    def get_value(self, x):
        return self.critic(self.forward(x))
