import numpy as np
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from numpy.random import choice
import time

action_map=["hp=1","hp=2","hp=3","hp=4","hp>=5","res=1","res=2","res=3"]


env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(1)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)


def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits / sum(logits))


env.action_space.seed(0)
obs = env.reset()
nvec = env.action_space.nvec
for i in range(10000):
    env.render()
    actions = []
    action_mask = np.array(env.vec_client.getMasks(0))[0]  # (16, 16, 79)
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])  # (256, 79)
    source_unit_mask = action_mask[:, [0]]  # (256, 1)
    source1 = (0, 0)
    source2 = (0, 1)
    base1 = (2, 2)
    for source_unit in np.where(source_unit_mask == 1)[0]:
        atpm = action_mask[source_unit, 1:]  # action_type_parameter_mask (78,)
        action_type = atpm[0:6]
        move_parameter = atpm[6:10]
        harvest_parameter = atpm[10:14]
        return_parameter = atpm[14:18]
        produce_direction_parameter = atpm[18:22]
        produce_unit_type_parameter = atpm[22:29]
        attack_target_parameter = atpm[29:sum(env.action_space.nvec[1:])]
        action = [
            source_unit,
            sample(atpm[0:6]),  # action type
            sample(atpm[6:10]),  # move parameter
            sample(atpm[10:14]),  # harvest parameter
            sample(atpm[14:18]),  # return parameter
            sample(atpm[18:22]),  # produce_direction parameter
            sample(atpm[22:29]),  # produce_unit_type parameter
            sample(atpm[29:sum(env.action_space.nvec[1:])]),  # attack_target parameter
        ]
        # state 1：if worker near resource, move to resource and harvest
        x = source_unit % 16
        y = source_unit // 16
        if atpm[5] == 1:
            action[5] = 2
        elif obs[0].reshape(-1, 27)[source_unit][17] == 1 and obs[0].reshape(-1, 27)[source_unit][5] == 1:
            if atpm[2] == 1:
                action[1] = 2
            else:
                action[1] = 1

        elif obs[0].reshape(-1, 27)[source_unit][17] == 1 and obs[0].reshape(-1, 27)[source_unit][6] == 1:
            if atpm[3] == 1:
                action[1] = 3
            else:
                action[1] = 1
                if base1[0] > x:
                    if move_parameter[1] == 1:
                        action[2] = 1
                if base1[1] > y:
                    if move_parameter[2] == 1:
                        action[2] = 2

        actions.append(action)
    next_obs, reward, done, info = env.step([actions])
    time.sleep(0.1)
env.close()
