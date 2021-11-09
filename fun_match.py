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
import seaborn as sns
import sys
from numpy.random import choice


def get_units_number(unit_type, bef_obs, ind_obs):
    return int(bef_obs.permute((0, 3, 1, 2))[ind_obs][unit_type].sum())


# base barracks worker light heavy range
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


def goal_result(gs):
    res = "base:" + str(gs[0][0]) + " barracks:" + str(gs[0][1]) + " worker:" + str(gs[0][2]) + " light:" + str(
        gs[0][3]) + " heavy:" + str(gs[0][4]) + " range" + str(gs[0][5])
    res = res + " op_base:" + str(gs[0][6]) + " op_barracks:" + str(gs[0][7]) + " op_worker:" + str(
        gs[0][8]) + " op_light:" + str(
        gs[0][9]) + " op_heavy:" + str(gs[0][10]) + " op_range" + str(gs[0][11])
    return res


def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits/sum(logits))


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
path = './model/microrts_fun_'+data+'_10_worker.pth'
path_pt = './model/microrts_fun_'+data+'_10_worker.pt'
path_manager = './model/microrts_fun_'+data+'_10_manager.pth'
path_pt_manager = './model/microrts_fun_'+data+'_10_manager.pt'
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
    map_path="maps/10x10/basesWorkers10x10.xml",
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

c = 16
inner_rewards = []

build_barracks = np.zeros(2000)
build_worker = np.zeros(2000)
build_light = np.zeros(2000)
build_range = np.zeros(2000)
build_heavy = np.zeros(2000)
return_resource = np.zeros(2000)
attack = np.zeros(2000)
move = np.zeros(2000)
attack_pos = np.zeros((10, 10))
move_pos = np.zeros((10, 10))
op_attack_pos = np.zeros((10, 10))
op_move_pos = np.zeros((10, 10))
s = 0
all_rewards = 0

for games in range(100):
    for step in range(2048):
        action_unit_mask = np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1)[0]
        selected_unit = sample(action_unit_mask)
        action_mask = np.array(envs.vec_client.getUnitActionMasks(np.array([selected_unit]))).reshape(1, -1)
        s = s+1
        envs.render()
        obs = next_obs
        if step % c == 0 or step == 0:
            with torch.no_grad():
                current_goal, manager_log_probs, _ = manager.get_goal(embedding(obs,1), num_envs)
                current_goal = torch.round(current_goal)
        with torch.no_grad():
            action, _, _, mask = worker.get_action(obs, current_goal, num_envs, envs=envs)
            action = action.T
        action = torch.Tensor([[12,0,3,3,1,1,3,12]]).int().to(device)
        next_obs, rs, ds, infos = envs.step(action)
        print(rs)
        a = action.cpu().numpy()
        pos = a[0][0]
        units_mask = mask[0][0:100]
        if units_mask[pos] == 1:
            pos_x = pos % 10
            pos_y = pos // 10
            if a[0][1] == 1:
                move_pos[pos_x][pos_y] = move_pos[pos_x][pos_y] + 1
                move[step] = move[step] + 1
            if a[0][1] == 3:
                return_resource[step] = return_resource[step] + 1
            if a[0][1] == 4:
                if a[0][6] == 2:
                    build_barracks[step] = build_barracks[step] + 1
                if a[0][6] == 3:
                    build_worker[step] = build_worker[step] + 1
                if a[0][6] == 4:
                    build_light[step] = build_light[step] + 1
                if a[0][6] == 6:
                    build_range[step] = build_range[step] + 1
                if a[0][6] == 5:
                    build_heavy[step] = build_heavy[step] + 1
            if a[0][1] == 5:
                attack_pos[pos_x][pos_y] = attack_pos[pos_x][pos_y] + 1
                attack[step] = attack[step] + 1
        if ds[0]:
            break
    print("\r", end="")
    print("Progress: {}%: ".format(games), "▋" * (games // 2), end="")
    sys.stdout.flush()
print('build worker: '+str(build_worker.sum()/s))
print('build light: '+str(build_light.sum()/s))
print('build range: '+str(build_range.sum()/s))
print('build heavy: '+str(build_heavy.sum()/s))
print('return resource: '+str(return_resource.sum()/s))
print('build barracks: '+str(build_barracks.sum()/s))
print('attack: '+str(attack.sum()/s))
print('move: '+str(move.sum()/s))

sns.heatmap(attack_pos/100, cmap='Reds')
plt.title('attack positions '+tittle)
plt.show()

sns.heatmap(move_pos/100, cmap='Blues')
plt.title('move positions '+tittle)
plt.show()

fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.scatter(range(2000), build_worker/100, color='blue', label='build worker', s=5)
ax1.scatter(range(2000), build_light/100, color='green', label='build light', s=5)
ax1.scatter(range(2000), build_range/100, color='yellow', label='build range', s=5)
ax1.scatter(range(2000), build_heavy/100, color='gray', label='build heavy', s=5)
ax1.scatter(range(2000), return_resource/100, color='cyan', label='return resource', s=5)
# ax1.scatter(range(2000), build_barracks/100, color='red', label='build barracks', s=5)
plt.xlabel('step')
plt.ylabel('action number')
plt.title('action distributions of 100 games '+tittle)
ax2 = plt.subplot(2, 1, 2)
ax2.scatter(range(2000), move/100, marker='p', color='green', label='move', s=5)
ax2.scatter(range(2000), attack/100, marker='o', color='blue', label='attack', s=5)
plt.xlabel('step')
plt.ylabel('action number')
ax1.set_ylim(bottom=0.005)
ax2.set_ylim(bottom=0.005)
ax1.legend()
ax2.legend()
plt.show()

print('end')


