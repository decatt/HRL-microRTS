import torch.nn as nn
import torch
import numpy as np
from mask import CategoricalMasked

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Worker(nn.Module):
    def __init__(self, envs):
        super(Worker, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(80, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(256, envs.action_space.nvec[1:].sum()), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x)

    def get_action(self, x, num_envs, selected_unit, action=None, invalid_action_masks=None, envs=None):
        logits = self.actor(self.forward(x))
        logits = torch.cat((selected_unit.to(device), logits), dim=1)
        split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

        if action is None:
            source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1))
            multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
            action_components = [multi_categoricals[0].sample()]
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

    def get_determinate_action(self, x, num_envs, selected_unit, envs=None):
        logits = self.actor(self.forward(x))
        logits = torch.cat((selected_unit.to(device), logits), dim=1)
        split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

        source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1))
        multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
        action_components = [multi_categoricals[0].sample()]
        source_unit_action_mask = torch.Tensor(
            np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(num_envs,
                                                                                                     -1))
        split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)

        action = torch.zeros((8, num_envs)).to(device)
        action[0] = selected_unit.argmax(dim=1).to(device)
        for i in range(7):
            action[i+1] = (torch.where(split_suam[i].to(device) != 0, split_logits[i+1], torch.tensor(-1e+8).to(device))).argmax(dim=1)

        return action

    def get_value(self, x):
        return self.critic(self.forward(x))


class Supervisor(nn.Module):
    def __init__(self):
        super(Supervisor, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(3, 3))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 3 * 3, 64)),
            nn.ReLU(), )
        self.actor = layer_init(nn.Linear(64, 4), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))

    def get_value(self, x):
        return self.critic(self.forward(x))

    def get_action(self, x, action=None, masks=None):
        logits = self.actor(self.forward(x))
        logits = torch.where(masks.to(device) != 0, logits, torch.tensor(-1e+8).to(device))
        c = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = c.sample()
        log_prob = c.log_prob(action)
        entropy = -(c.logits*c.probs).sum(-1)
        return action, log_prob, entropy, masks


class SelfPlayWorker(nn.Module):
    def __init__(self, envs):
        super(SelfPlayWorker, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(54, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(256, envs.action_space.nvec[1:].sum()), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x)

    def get_action(self, x, selected_units, source_unit_action_mask, action=None, envs=None):
        logits = self.actor(self.forward(x))
        logits = torch.cat((selected_units.to(device), logits), dim=1)
        split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)
        invalid_action_masks = torch.cat((selected_units, source_unit_action_mask), 1)
        if action is None:
            # 1. select source unit based on source unit mask
            multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=selected_units)]
            action_components = [multi_categoricals[0].sample()]
            # 2. select action type and parameter section based on the
            #    source-unit mask of action type and parameters

            split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
            multi_categoricals = multi_categoricals + [
                CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)
            ]
            action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
            action = torch.stack(action_components)
        else:
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                zip(split_logits, split_invalid_action_masks)
            ]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        logprob = logprob.T.view(-1, 10, 8)
        entropy = entropy.T.view(-1, 10, 8)
        action = action.T.view(-1, 10, 8)
        return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

    def get_value(self, x):
        return self.critic(self.forward(x))