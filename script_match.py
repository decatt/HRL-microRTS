import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from env_wrapper import VecMonitor, VecPyTorch, MicroRTSStatsRecorder
import time
from numpy.random import choice
import lower_level_script
from dimension_reduction import dict_graph, get_node_vector


def get_units_number(unit_type, bef_obs, ind_obs):
    return int(bef_obs.permute((0, 3, 1, 2))[ind_obs][unit_type].sum())


def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits / sum(logits))


gamma = 0.99

seed = 0

num_envs = 1

# build worker 1 090604
# resource 1 090611
# attack 1 090619
# build barracks units 090701

tittle = 'reward: normal'
data = '2021092806'

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
path = './model/microrts_fun_' + data + '_10_worker.pth'
path_pt = './model/microrts_fun_' + data + '_10_worker.pt'
path_manager = './model/microrts_fun_' + data + '_10_manager.pth'
path_pt_manager = './model/microrts_fun_' + data + '_10_manager.pt'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ais = []
for i in range(num_envs):
    ais.append(microrts_ai.workerRushAI)

envs = MicroRTSVecEnv(
    num_envs=num_envs,
    max_steps=2000,
    render_theme=2,
    ai2s=ais,
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
envs = MicroRTSStatsRecorder(envs, gamma)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)
next_obs = envs.reset()

for games in range(100):
    for step in range(2048):
        envs.render()
        obs = next_obs
        actions= []
        for i in range(num_envs):
            gs = obs[i]
            action_unit_mask = np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)[i]
            if np.sum(action_unit_mask) > 1 and action_unit_mask[34] == 1:
                action_unit_mask[34] = 0
            selected_unit = sample(action_unit_mask)
            action_mask = np.array(envs.vec_client.getUnitActionMasks(np.array([selected_unit]))).reshape(num_envs, -1)[i]
            target = (14, 14)
            action, target = lower_level_script.harvest_resource(gs, selected_unit, action_mask, target)
            action = lower_level_script.barracks_produce(action, 6)
            actions.append(action)
        actions = torch.Tensor(actions).int().to(device)
        next_obs, _, _, _ = envs.step(actions)
        time.sleep(0.02)
