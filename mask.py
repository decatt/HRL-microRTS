import torch
from torch.distributions.categorical import Categorical

use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    # H = sum(p(x)log(p(x)))
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

    def deterministic(self):
        p_log_p = self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        # res = torch.tensor([torch.tensor([torch.argmax(distribution) for distribution in p_log_p])])
        res = []
        for distribution in p_log_p:
            d = torch.argmax(distribution)
            res.append(d)
        res = torch.tensor(res)
        return res


def gae_adv(gae, rewards, num_steps, next_done, last_value, dones, values):
    gamma = 0.99
    gae_lambda = 1
    if gae:
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        returns = advantages + values
    else:
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_return = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + gamma * next_non_terminal * next_return
            advantages = returns - values
    return returns
