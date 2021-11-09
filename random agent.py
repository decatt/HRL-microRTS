import numpy as np
from numpy.random import choice
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import time

env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(1)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
# env = VecVideoRecorder(env, 'videos', record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)

def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits/sum(logits))

env.action_space.seed(0)
env.reset()
nvec = env.action_space.nvec

build_barracks = np.zeros(2000)
build_worker = np.zeros(2000)
build_light = np.zeros(2000)
build_range = np.zeros(2000)
build_heavy = np.zeros(2000)
return_resource = np.zeros(2000)
attack = np.zeros(2000)
move = np.zeros(2000)
attack_pos = np.zeros((16, 16))
move_pos = np.zeros((16, 16))
op_attack_pos = np.zeros((16, 16))
op_move_pos = np.zeros((16, 16))
s = 0
all_rewards = 0
tittle = 'random'

for games in range(100):
    for step in range(2000):
        s = s + 1
        env.render()
        action = []
        action_mask = np.array(env.vec_client.getMasks(0))[0] # (16, 16, 79)
        action_mask = action_mask.reshape(-1, action_mask.shape[-1]) # (256, 79)
        source_unit_mask = action_mask[:,[0]] # (256, 1)

        source_unit = sample(source_unit_mask.reshape(-1))
        atpm = action_mask[source_unit,1:] # action_type_parameter_mask (78,)
        action += [[
            source_unit,
            sample(atpm[0:6]), # action type
            sample(atpm[6:10]), # move parameter
            sample(atpm[10:14]), # harvest parameter
            sample(atpm[14:18]), # return parameter
            sample(atpm[18:22]), # produce_direction parameter
            sample(atpm[22:29]), # produce_unit_type parameter
            sample(atpm[29:sum(env.action_space.nvec[1:])]), # attack_target parameter
        ]]
        next_obs, reward, done, info = env.step([action])
        if done[0]:
            break

        if len(action) > 0:
            a = action
            pos = a[0][0]
            pos_x = pos % 16
            pos_y = pos // 16
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
        if done[0]:
            break
    print("\r", end="")
    print("Download progress: {}%: ".format(games), "▋" * (games // 2), end="")
    sys.stdout.flush()
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
ax1.scatter(range(2000), build_worker / 100, color='blue', label='build worker', s=5)
ax1.scatter(range(2000), build_light / 100, color='green', label='build light', s=5)
ax1.scatter(range(2000), build_range / 100, color='yellow', label='build range', s=5)
ax1.scatter(range(2000), build_heavy / 100, color='gray', label='build heavy', s=5)
ax1.scatter(range(2000), return_resource / 100, color='cyan', label='return resource', s=5)
# ax1.scatter(range(2000), build_barracks/100, color='red', label='build barracks', s=5)
plt.xlabel('step')
plt.ylabel('action number')
plt.title('action distributions of 100 games ' + tittle)
ax2 = plt.subplot(2, 1, 2)
ax2.scatter(range(2000), move / 100, marker='p', color='green', label='move', s=5)
ax2.scatter(range(2000), attack / 100, marker='o', color='blue', label='attack', s=5)
plt.xlabel('step')
plt.ylabel('action number')
ax1.set_ylim(bottom=0.005)
ax2.set_ylim(bottom=0.005)
ax1.legend()
ax2.legend()
plt.show()

print('end')
