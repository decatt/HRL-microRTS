import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from numpy.random import choice


def sample(logits):
    # sample 1 or 2 from logits [0, 1 ,1, 0] but not 0 or 3
    if sum(logits) == 0: return 0
    return choice(range(len(logits)), p=logits/sum(logits))


if __name__ == "__main__":
    use_gpu = True
    seed = 0
    num_envs = 1
    map_size = 10

    if map_size == 16:
        map_path = "maps/16x16/basesWorkers16x16.xml"
    elif map_size == 10:
        map_path = "maps/10x10/basesWorkers10x10.xml"

    envs = MicroRTSVecEnv(
        num_envs=num_envs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.naiveMCTSAI for _ in range(num_envs)],
        map_path="maps/16x16/basesWorkers16x16.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    obs = envs.reset()
    nvec = envs.action_space.nvec
    for i in range(10000):
        envs.render()
        actions = []
        action_mask = np.array(envs.vec_client.getMasks(0))[0]  # (16, 16, 79)
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])  # (256, 79)
        source_unit_mask = action_mask[:, [0]]  # (256, 1)
        for source_unit in np.where(source_unit_mask == 1)[0]:
            atpm = action_mask[source_unit, 1:]  # action_type_parameter_mask (78,)
            actions += [[
                source_unit,
                sample(atpm[0:6]),  # action type
                sample(atpm[6:10]),  # move parameter
                sample(atpm[10:14]),  # harvest parameter
                sample(atpm[14:18]),  # return parameter
                sample(atpm[18:22]),  # produce_direction parameter
                sample(atpm[22:29]),  # produce_unit_type parameter
                sample(atpm[29:sum(envs.action_space.nvec[1:])]),  # attack_target parameter
            ]]
        next_obs, reward, done, info = envs.step([actions])
