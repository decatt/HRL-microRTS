import numpy as np
import torch
from gym_microrts.envs.vec_env import MicroRTSVecEnv


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
            d = abs(units[i].x - units[j].x) + abs(units[i].y-units[j].y)
            graph[i][j] = d
    return graph, units


def dict_graph(gs:torch.Tensor, size: int):
    graph = dict()
    units = (torch.ones((size, size)).to(device)-gs.permute((2, 0, 1))[10]).reshape(-1)
    g = gs.reshape(-1,27)
    for pos in range(size*size):
        if units[pos] == 1:
            graph[pos] = g[pos]
    return graph


def get_node_vector(node: int, state_graph: dict, size: int):
    res = torch.zeros((27,)).to(device)
    for key in state_graph.keys():
        w = 0
        if key != node:
            key_x = key % size
            key_y = key//size
            node_x = node % size
            node_y = node//size
            w = 1/(abs(key_x - node_x) + abs(key_y - node_y))
        res = res + w*state_graph[key]
    return res


def get_selected_node_vector(node, gs, size):
    v1 = torch.zeros(27,).to(device)
    node_x = node % size
    node_y = node // size
    units = (torch.ones((size, size)).to(device) - gs.permute((2, 0, 1))[10]).reshape(-1)
    for pos in range(size*size):
        if pos != node and units[pos] == 1:
            pos_x = pos % size
            pos_y = pos // size
            w = 1 / (abs(pos_x - node_x) + abs(pos_y - node_y))
            v1 = v1 + w*gs[pos_y][pos_x]
    return torch.cat((v1, gs[node_y][node_x]))
