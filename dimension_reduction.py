import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def count_units(states: torch.Tensor, num_env: int) -> torch.Tensor:
    zs = torch.zeros((num_env, 12))
    n = 0
    for state in states:
        z = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        z[0] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[15]).sum()
        z[1] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[16]).sum()
        z[2] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[17]).sum()
        z[3] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[18]).sum()
        z[4] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[19]).sum()
        z[5] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[20]).sum()
        z[6] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[15]).sum()
        z[7] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[16]).sum()
        z[8] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[17]).sum()
        z[9] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[18]).sum()
        z[10] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[19]).sum()
        z[11] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[20]).sum()
        zs[n] = z
        n = n + 1
    return zs


def position(state: torch.Tensor):
    return state.permute((2, 0, 1))[9:20]


class Unit:
    def __init__(self, x: int, y: int, unit_type: str):
        self.x = x
        self.y = y
        self.unit_type = unit_type


def position_graph(state: torch.Tensor, size: int):
    units = dict()
    index = 0
    for i in range(15, 21):
        z = state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[i]
        t = ''
        if i == 16:
            t = 'base'
        if i == 17:
            t = 'worker'
        if i == 18:
            t = 'barracks'
        if i == 19:
            t = 'light'
        if i == 20:
            t = 'heavy'
        if i == 21:
            t = 'light'
        for x in range(size):
            for y in range(size):
                if z[x][y] == 1:
                    units[index] = Unit(x, y, t)
                    index = index + 1
    graph = torch.zeros((index, index))
    for i in range(index):
        for j in range(index):
            # Manhattan Distance
            d = abs(units[i].x - units[j].x) + abs(units[i].y - units[j].y)
            graph[i][j] = d
    return graph, units


def dict_graph(gs: torch.Tensor, size: int):
    graph = dict()
    units = (torch.ones((size, size)).to(device) - gs.permute((2, 0, 1))[10]).reshape(-1)
    g = gs.reshape(-1, 27)
    for pos in range(size * size):
        if units[pos] == 1:
            graph[pos] = g[pos]
    return graph


def get_node_vector(node: int, state_graph: dict, size: int):
    res = torch.zeros((27,)).to(device)
    for key in state_graph.keys():
        w = 0
        if key != node:
            key_x = key % size
            key_y = key // size
            node_x = node % size
            node_y = node // size
            w = 1 / (abs(key_x - node_x) + abs(key_y - node_y))
        res = res + w * state_graph[key]
    return res


def get_selected_node_vector(node, gs, size):
    v1 = torch.zeros(27, ).to(device)
    v2 = torch.zeros(26, ).to(device)
    node_x = node % size
    node_y = node // size
    units = (torch.ones((size, size)).to(device) - gs.permute((2, 0, 1))[10]).reshape(-1)
    for pos in range(size * size):
        if pos != node and units[pos] == 1:
            pos_x = pos % size
            pos_y = pos // size
            d = abs(pos_x - node_x) + abs(pos_y - node_y)
            inf = gs[pos_y][pos_x]
            w = 1 / d
            v1 = v1 + w * inf

            v2 = v2 + directions(pos_x, pos_y, node_x, node_y, d, inf)
    return torch.cat((v2, v1, gs[node_y][node_x]))


def get_round_imf(size, map_size, pos, game_state):
    res = torch.zeros((2 * size + 1, 2 * size + 1, 27))
    x = pos % map_size
    y = pos//map_size
    for i in range(2 * size + 1):
        for j in range(2 * size + 1):
            if 0 <= x - size + i < map_size and 0 <= y - size + j < map_size:
                res[j][i] = game_state[y - size + j][x - size + i]
    return res


def directions(pos_x, pos_y, node_x, node_y, d, inf):
    ind = -1
    if inf[14] == 1:
        ind = 0
    elif inf[11] == 1:
        if inf[15] == 1:
            ind = 1
        if inf[16] == 1:
            ind = 2
        if inf[17] == 1:
            ind = 3
        if inf[18] == 1:
            ind = 4
        if inf[19] == 1:
            ind = 5
        if inf[20] == 1:
            ind = 6
    elif inf[12] == 1:
        if inf[15] == 1:
            ind = 7
        if inf[16] == 1:
            ind = 8
        if inf[17] == 1:
            ind = 9
        if inf[18] == 1:
            ind = 10
        if inf[19] == 1:
            ind = 11
        if inf[20] == 1:
            ind = 12
    res = torch.zeros((26,)).to(device)
    x = (pos_x - node_x)/d
    y = (pos_y - node_y)/d
    res[2*ind] = x
    res[2*ind+1] = y
    return res


def rotate_action(action, rotate_type):
    if rotate_type == "":
        action[1] = action([3, 0, 1, 2])
    return action
