import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from env_wrapper import VecMonitor, VecPyTorch, MicroRTSStatsRecorder
from GNNAgent import Worker, Supervisor
from numpy.random import choice
from dimension_reduction import get_selected_node_vector, get_round_imf
from lower_level_script import force_harvest_resource
import seaborn as sns
import sys
import matplotlib.pyplot as plt


def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits / sum(logits))


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

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
# path = './model/microrts_ppo_' + data + '_16_worker.pth'
path_pt = './model/microrts_GNN_ppo_2021112100_10_worker.pt'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 2021112109
# tiamat 0.79
# coach AI 0.04
# worker rush 0.36


ais = []
for i in range(num_envs):
    ais.append(microrts_ai.naiveMCTSAI)

size = 8
map_path = "maps/8x8/basesWorkers8x8.xml"
if size == 10:
    map_path = "maps/10x10/basesWorkers10x10.xml"
elif size == 16:
    map_path = "maps/16x16/basesWorkers16x16.xml"

init_seeds()
envs = MicroRTSVecEnv(
    num_envs=num_envs,
    max_steps=1500,
    render_theme=2,
    ai2s=ais,
    frame_skip=10,
    map_path=map_path,
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
envs = MicroRTSStatsRecorder(envs, gamma)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)
next_obs = envs.reset()

worker = Worker(envs).to(device)

worker.load_state_dict(torch.load(path_pt, map_location=device))
# sup = Supervisor().to(device)

# sup.load_state_dict(torch.load('./model/microrts_GNNppo_2021111707_10_sup.pt', map_location=device))

all_rewards = 0

build_barracks = np.zeros(5000)
build_worker = np.zeros(5000)
build_light = np.zeros(5000)
build_range = np.zeros(5000)
build_heavy = np.zeros(5000)
return_resource = np.zeros(5000)
attack = np.zeros(5000)
move = np.zeros(5000)
attack_pos = np.zeros((size, size))
move_pos = np.zeros((size, size))
op_attack_pos = np.zeros((size, size))
op_move_pos = np.zeros((size, size))
s = 0

tittle = 'GNN against worker rush in ' + str(size) + 'x' + str(size)

outcomes = []

for games in range(100):
    for step in range(5000):
        s = s + 1
        envs.render()
        obs = next_obs
        graphs = torch.zeros((num_envs, 80))
        selected_units = torch.zeros((num_envs, size * size))
        round_imfs = torch.zeros((num_envs, 7, 7, 27))
        masks = torch.zeros((num_envs, 4))
        force_action = False
        for i in range(num_envs):
            action_unit_mask = np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)[i]
            if sum(action_unit_mask) > 0:
                selected_unit = sample(action_unit_mask)
                graphs[i] = get_selected_node_vector(selected_unit, next_obs[i], size)
            else:
                selected_unit = 0
                graphs[i] = torch.zeros((80,))
            selected_units[i][selected_unit] = 1
            round_imf = get_round_imf(3, size, selected_unit, next_obs[i])
            round_imfs[0] = round_imf

        if not force_action:
            with torch.no_grad():
                # action = worker.get_determinate_action(graphs.to(device), num_envs, selected_units, envs=envs)
                action, _, _, mask = worker.get_action(graphs.to(device), num_envs, selected_units, envs=envs)
        # d, _, _, _ = sup.get_action(round_imfs.to(device), masks=mask.permute((1, 0))[106:110].permute((1, 0)))
        # action[2] = d.T
        # mask = torch.zeros((num_envs, size*size))
        action = action.T

        next_obs, rs, ds, infos = envs.step(action)
        a = action.cpu().numpy()
        pos = a[0][0]
        units_mask = mask[0][0:size * size]
        if units_mask[pos] == 1:
            pos_x = pos % size
            pos_y = pos // size
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
            if get_units_number(11, obs, 0) > get_units_number(12, obs, 0):
                outcomes.append(1)
            else:
                outcomes.append(0)
            break
    print("\r", end="")
    print("Progress: {}%: ".format(games), "▋" * (games // 2), end="")
    sys.stdout.flush()
print(sum(outcomes) / len(outcomes))
print('build worker: ' + str(build_worker.sum() / s))
print('build light: ' + str(build_light.sum() / s))
print('build range: ' + str(build_range.sum() / s))
print('build heavy: ' + str(build_heavy.sum() / s))
print('return resource: ' + str(return_resource.sum() / s))
print('build barracks: ' + str(build_barracks.sum() / s))
print('attack: ' + str(attack.sum() / s))
print('move: ' + str(move.sum() / s))

sns.heatmap(attack_pos / 100, cmap='Reds')
plt.title('attack positions ' + tittle)
plt.show()

sns.heatmap(move_pos / 100, cmap='Blues')
plt.title('move positions ' + tittle)
plt.show()

fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.scatter(range(5000), build_worker / 100, color='blue', label='build worker', s=5)
ax1.scatter(range(5000), build_light / 100, color='green', label='build light', s=5)
ax1.scatter(range(5000), build_range / 100, color='yellow', label='build range', s=5)
ax1.scatter(range(5000), build_heavy / 100, color='gray', label='build heavy', s=5)
ax1.scatter(range(5000), return_resource / 100, color='cyan', label='return resource', s=5)
# ax1.scatter(range(2000), build_barracks/100, color='red', label='build barracks', s=5)
plt.xlabel('step')
plt.ylabel('action number')
plt.title('action distributions of 100 games ' + tittle)
ax2 = plt.subplot(2, 1, 2)
ax2.scatter(range(5000), move / 100, marker='p', color='green', label='move', s=5)
ax2.scatter(range(5000), attack / 100, marker='o', color='blue', label='attack', s=5)
plt.xlabel('step')
plt.ylabel('action number')
ax1.set_ylim(bottom=0.005)
ax2.set_ylim(bottom=0.005)
ax1.legend()
ax2.legend()
plt.show()

print('end')
