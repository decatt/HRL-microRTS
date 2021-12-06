import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from env_wrapper import VecMonitor, VecPyTorch, MicroRTSStatsRecorder
from ppo_agent import Worker
from numpy.random import choice


def get_units_number(unit_type, bef_obs, ind_obs):
    return int(bef_obs.permute((0, 3, 1, 2))[ind_obs][unit_type].sum())


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

data = '2021090917'

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
path = './model/microrts_ppo_'+data+'_16_worker.pth'
path_pt = './model/microrts_ppo_'+data+'_16_worker.pt'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ais = []
for i in range(num_envs):
    ais.append(microrts_ai.coacAI)

init_seeds()
envs = MicroRTSVecEnv(
    num_envs=num_envs,
    max_steps=5000,
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

worker = Worker(envs).to(device)

worker.load_state_dict(torch.load(path_pt, map_location=device))

inner_rewards = []

all_rewards = 0
for games in range(100):
    for step in range(5000):
        envs.render()
        obs = next_obs
        with torch.no_grad():
            action, _, _, mask = worker.get_action(obs, num_envs, envs=envs)
            action = action.T

        next_obs, rs, ds, infos = envs.step(action)
        if ds[0]:
            break
print('end')
