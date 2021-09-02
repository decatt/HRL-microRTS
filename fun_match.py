import numpy
import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from env_wrapper import VecMonitor, VecPyTorch, MicroRTSStatsRecorder
import torch.optim as optim
from feudal_agent import Manager, Worker
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import collections


def get_units_number(unit_type, bef_obs, ind_obs):
    return int(bef_obs.permute((0, 3, 1, 2))[ind_obs][unit_type].sum())


def embedding(states, num_env):
    zs = torch.zeros((num_env, 12)).to(device)
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


# cos_sim
def get_inner_reward(goal, v_zs, v_zs_, num_env):
    res = torch.zeros((num_env,))
    for i_env in range(num_env):
        vector_a = goal[i_env] - v_zs[i_env]
        vector_b = goal[i_env] - v_zs_[i_env]
        num = torch.mm(vector_a.view((1, 12)), vector_b.view((12, 1)))
        denom = torch.norm(vector_a) * torch.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        res[i_env] = sim[0]*torch.norm(v_zs_[i_env] - v_zs[i_env])
    return res


def init_seeds(torch_seed=0, seed=0):
    torch.manual_seed(torch_seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(torch_seed)  # Sets the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed_all(torch_seed)  # Sets the seed for generating random numbers on all GPUs.
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


gamma = 0.99

seed = 0

num_envs = 1

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
path = './model/microrts_fun_0_16_worker_hp.pth'
path_pt = './model/microrts_fun_0901_16_worker_hp.pt'
path_manager = './model/microrts_fun_0901_16_manager_hp.pth'
path_pt_manager = './model/microrts_fun_0901_16_manager_hp.pt'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ais = []
for i in range(num_envs):
    ais.append(microrts_ai.coacAI)

init_seeds()
envs = MicroRTSVecEnv(
    num_envs=num_envs,
    max_steps=2000,
    render_theme=2,
    ai2s=ais,
    frame_skip=10,
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
envs = MicroRTSStatsRecorder(envs, gamma)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)
next_obs = envs.reset()
manager = Manager().to(device)
worker = Worker(envs).to(device)

worker.load_state_dict(torch.load(path_pt, map_location=device))
manager.load_state_dict(torch.load(path_pt_manager, map_location=device))

c = 64

for step in range(1000000):
    envs.render()
    if step % c == 0 or step == 0:
        with torch.no_grad():
            current_goal, manager_log_probs, _ = manager.get_goal(next_obs, num_envs)

    with torch.no_grad():
        action, _, _, _ = worker.get_action(next_obs, current_goal, num_envs, envs=envs)
        next_obs, rs, ds, infos = envs.step(action.T)

