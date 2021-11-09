import numpy as np
import torch
from numpy.random import choice
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0:
        return 0
    return choice(range(len(logits)), p=logits / sum(logits))


def near_target(unit_pos_x: int, unit_pos_y: int, target_x: int, target_y: int) -> bool:
    if unit_pos_x + 1 == target_x and unit_pos_y == target_y:
        return True
    if unit_pos_x - 1 == target_x and unit_pos_y == target_y:
        return True
    if unit_pos_x == target_x and unit_pos_y + 1 == target_y:
        return True
    if unit_pos_x == target_x and unit_pos_y - 1 == target_y:
        return True
    return False


def get_default_action(unit, action_temp):
    action = [
        unit,
        sample(action_temp[0:6]),  # action type: NOOP, move, harvest, return, produce, attack
        sample(action_temp[6:10]),  # move parameter
        sample(action_temp[10:14]),  # harvest parameter
        sample(action_temp[14:18]),  # return parameter
        sample(action_temp[18:22]),  # produce_direction parameter
        sample(action_temp[22:29]),  # produce_unit_type parameter
        sample(action_temp[29:78]),  # attack_target parameter
    ]
    return action


def move_to_target(x, y, target, action, action_temp):
    move_parameter = action_temp[6:10]
    if action_temp[1] == 1:
        action[1] = 1
        if target[0] > x and move_parameter[1] == 1:
            action[2] = 1
        if target[1] > y and move_parameter[2] == 1:
            action[2] = 2
        if target[0] < x and move_parameter[3] == 1:
            action[2] = 3
        if target[1] < y and move_parameter[0] == 1:
            action[2] = 0
    return action


def auto_attack(units: [int], action_mask):
    actions = []
    for unit in units:
        action_temp = action_mask[unit]
        action = get_default_action(unit, action_temp)
        if action_temp[5] == 1:
            action[1] = 5
        actions.append(action)
    return actions


def auto_return(units: [int], action_mask):
    actions = []
    for unit in units:
        action_temp = action_mask[unit]
        action = get_default_action(unit, action_temp)
        if action_temp[3] == 1:
            action[1] = 3
        actions.append(action)
    return actions


def harvest_resource(gs: torch.Tensor, worker: int, action_temp, target):
    workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17]
    barracks = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[16]
    resource = torch.ones((16, 16)).to(device) - gs.permute((2, 0, 1))[5]
    resource_workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17] * gs.permute((2, 0, 1))[6]
    worker_without_resource = workers - resource_workers
    # actions = []
    x = worker % 16
    y = worker // 16
    move_parameter = action_temp[6:10]
    action = get_default_action(worker, action_temp)
    if resource_workers[y][x] == 1:
        target = (2, 2)
    elif worker_without_resource[y][x] == 1 and x <= 2 and y <= 2:
        d1 = abs(x - 0) + abs(y - 0)
        d2 = abs(x - 0) + abs(y - 1)
        if d1 > d2 and resource[1][0] == 1:
            target = (0, 1)
        elif resource[0][0] == 1:
            target = (0, 0)
    if action_temp[5] == 1:  # attack
        action[1] = 5
    elif action_temp[3] == 1:  # return
        action[1] = 3
    elif action_temp[2] == 1 and x < 8 and y < 8:  # harvest
        action[1] = 2
    elif action_temp[4] == 1 and ((y, x) == (0, 2) or (y, x) == (0, 4) or (y, x) == (1, 3)) and barracks.sum() < 1:
        action[1] = 4
        if (y, x) == (0, 2):
            action[5] = 1
        if (y, x) == (0, 4):
            action[5] = 3
        if (y, x) == (1, 3):
            action[5] = 0
    elif action_temp[1] == 1 and workers[x][y] == 1:
        move_parameter = action_temp[6:10]
        action[1] = 1
        if target[0] > x and move_parameter[1] == 1:
            action[2] = 1
        if target[1] > y and move_parameter[2] == 1:
            action[2] = 2
        if target[0] < x and move_parameter[3] == 1:
            action[2] = 3
        if target[1] < y and move_parameter[0] == 1:
            action[2] = 0
    return action, target


def barracks_produce(action, unit_type):
    if action[1] == 4 and action[6] >= 4:
        action[6] = unit_type
        action[5] = 1
    return action


def move_to_region(region):
    pass


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HigherLevelNetwork(torch.nn.Module):
    def __init__(self):
        super(HigherLevelNetwork, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 6 * 6, 256)),
            # layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(), )

        self.actor = layer_init(nn.Linear(256, 8), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2)))


def force_harvest_resource(gs: torch.Tensor, worker: int, action_temp):
    force_action = True
    workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17]
    resource = torch.ones((16, 16)).to(device) - gs.permute((2, 0, 1))[5]
    resource_workers = gs.permute((2, 0, 1))[11] * gs.permute((2, 0, 1))[17] * gs.permute((2, 0, 1))[6]
    worker_without_resource = workers - resource_workers
    # actions = []
    x = worker % 16
    y = worker // 16
    if workers[y][x] ==1:
        action = get_default_action(worker, action_temp)
        target = (2, 2)
        if resource_workers[y][x] == 1:
            target = (2, 2)
        elif worker_without_resource[y][x] == 1 and x <= 2 and y <= 2:
            d1 = abs(x - 0) + abs(y - 0)
            d2 = abs(x - 0) + abs(y - 1)
            if d1 > d2 and resource[1][0] == 1:
                target = (0, 1)
            elif resource[0][0] == 1:
                target = (0, 0)
        if action_temp[5] == 1:  # attack
            action[1] = 5
        elif action_temp[3] == 1:  # return
            action[1] = 3
        elif action_temp[2] == 1 and x < 8 and y < 8:  # harvest
            action[1] = 2
        elif action_temp[1] == 1 and workers[x][y] == 1:
            move_parameter = action_temp[6:10]
            action[1] = 1
            if target[0] > x and move_parameter[1] == 1:
                action[2] = 1
            elif target[1] > y and move_parameter[2] == 1:
                action[2] = 2
            elif target[0] < x and move_parameter[3] == 1:
                action[2] = 3
            elif target[1] < y and move_parameter[0] == 1:
                action[2] = 0
        else:
            force_action = False
    else:
        force_action = False
        action = []
    return action, force_action
