import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from env_wrapper import VecMonitor , VecPyTorch, MicroRTSStatsRecorder
import torch.optim as optim
from GNNAgent import Worker
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
from dimension_reduction import dict_graph, get_node_vector, get_selected_node_vector
from numpy.random import choice
import gym


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


size = 16
test_ai = False
use_gpu = True
seed = 0
num_envs = 20
torch_seeds = 0
gamma = 0.99
learning_rate = 2.5e-4
num_steps = 1024
total_steps = 128000000
n_minibatch = 4
anneal = True
manager_learn = True

lr = lambda f: f * learning_rate
gae = True
update_epochs = 4
norm_adv = True
clip_coef = 0.1
clip_vloss = True
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5

batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // n_minibatch)

device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
date = time.strftime("%Y%m%d%H", time.localtime())

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ais = []
for i in range(num_envs):
    ais.append(microrts_ai.coacAI)

init_seeds()


############################################

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.int32,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


class NoAvailableActionThenSkipEnv(gym.Wrapper):
    """if no source unit can be selected in microrts,
    automatically execute a NOOP action
    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        while self.unit_location_mask.sum() == 0:
            obs, reward, done, info = self.env.step(action)
            if done:
                break
        return obs, reward, done, info


envs = MicroRTSVecEnv(
    num_envs=num_envs,
    max_steps=10000,
    render_theme=2,
    ai2s=ais,
    frame_skip=10,
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
# envs = VecEnvWrapper(envs)
envs = MicroRTSStatsRecorder(envs, gamma)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)

#############################################

worker = Worker(envs).to(device)
opponent_agent = Worker(envs).to(device)

global_step = 0
start_time = time.time()
next_obs = envs.reset()
next_done = torch.zeros(num_envs).to(device)
num_updates = total_steps // batch_size  # 128000000/8
starting_update = 1
gae_lambda = 0.95
worker_optimizer = optim.Adam(worker.parameters(), lr=learning_rate, eps=1e-5)

obs = torch.zeros((num_steps, num_envs, 54)).to(device)
all_selected_units = torch.zeros((num_steps, num_envs, size * size)).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.action_space.shape).to(device)
log_probs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)
invalid_action_masks = torch.zeros((num_steps, num_envs) + (envs.action_space.nvec.sum(),)).to(device)

step_rewards = collections.deque(maxlen=5000)
outcomes = collections.deque(maxlen=500)
outcomes_record = []
rewards_record = []

hp_outcome = 0.0

update_rewards = True
rewards_setting = [10.0, 1.0, 1.0, 0.2, 1.0, 4.0]



for update in range(starting_update, num_updates + 1):
    if anneal:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        worker_optimizer.param_groups[0]['lr'] = lrnow
    for step in range(0, num_steps):
        envs.render()
        global_step += 1 * num_envs
        graphs = torch.zeros((num_envs, 54))
        selected_units = torch.zeros((num_envs, size * size))

        for i in range(num_envs):
            action_unit_mask = np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)[i]
            if sum(action_unit_mask) > 0:
                selected_unit = sample(action_unit_mask)
                graphs[i] = get_selected_node_vector(selected_unit, next_obs[i], size)
            else:
                selected_unit = 0
                graphs[i] = torch.zeros((54,))
            selected_units[i][selected_unit] = 1
        bef_obs = next_obs
        all_selected_units[step] = selected_units
        obs[step] = graphs
        dones[step] = next_done

        o_obs = torch.Tensor(np.transpose(envs.venv.get_attr("opponent_obs"), axes=(0, 3, 1, 2))).to(device)
        o_graphs = torch.zeros((num_envs, 54))
        o_selected_units = torch.zeros((num_envs, size * size))
        invalid_action_masks = torch.Tensor(np.array(envs.venv.get_attr("opponent_action_mask")))
        for i in range(num_envs):

            o_action_unit_mask = np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)[i]
            if sum(o_action_unit_mask) > 0:
                o_selected_unit = sample(o_action_unit_mask)
                o_graphs[i] = get_selected_node_vector(o_selected_unit, o_obs[i], size)
            else:
                o_selected_unit = 0
                o_graphs[i] = torch.zeros((54,))
            o_selected_units[i][o_selected_unit] = 1
        with torch.no_grad():
            o_action, _, _, _ = worker.get_action(o_graphs, num_envs, o_selected_units, envs=envs)
            for i in range(envs.num_envs):
                envs.venv.env_method("set_opponent_action", o_action.T[i].tolist(), indices=i)

        with torch.no_grad():
            values[step] = worker.get_value(obs[step]).flatten()

            action, log_prob, _, invalid_action_masks[step] = worker.get_action(obs[step], num_envs, selected_units, envs=envs)
            actions[step] = action.T
            log_probs[step] = log_prob
            next_obs, rs, ds, infos = envs.step(action.T)
            for index in range(num_envs):
                step_rewards.append(rs[index].numpy()[0])
                if ds[index]:
                    if get_units_number(11, bef_obs, index) > get_units_number(12, bef_obs, index):
                        outcomes.append(1)
                    else:
                        outcomes.append(0)

        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
    if len(outcomes) > 0:
        outcomes_record.append(sum(outcomes) / len(outcomes))
        rewards_record.append(sum(step_rewards) / len(step_rewards))

    with torch.no_grad():
        last_value = worker.get_value(obs[-1].to(device)).reshape(1, -1)
        if gae:
            advantages = torch.zeros_like(rewards).to(device)

            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_return = last_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * next_non_terminal * next_return
                advantages = returns - values

    b_obs = obs.reshape((-1,) + (54,))
    b_log_probs = log_probs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.action_space.shape)
    b_selected_units = all_selected_units.reshape((-1,) + (size * size,))
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1, invalid_action_masks.shape[-1]))
    # worker learning
    target_worker = Worker(envs).to(device)
    inds = np.arange(batch_size, )
    for i_epoch_pi in range(update_epochs):
        np.random.shuffle(inds)
        target_worker.load_state_dict(worker.state_dict())
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, new_log_prob, entropy, _ = worker.get_action(
                b_obs[minibatch_ind],
                num_envs=num_envs,
                selected_unit=b_selected_units[minibatch_ind],
                action=b_actions.long()[minibatch_ind].T,
                invalid_action_masks=b_invalid_action_masks[minibatch_ind],
                envs=envs)
            ratio = (new_log_prob - b_log_probs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_log_probs[minibatch_ind] - new_log_prob).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = worker.get_value(b_obs[minibatch_ind]).view(-1)
            if clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -clip_coef,
                                                                  clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            worker_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(worker.parameters(), max_grad_norm)
            worker_optimizer.step()

    if update % 64 == 0:
        date = time.strftime("%Y%m%d%H", time.localtime())
        path = './model/microrts_GNNppo_' + date + '_10_worker.pth'
        path_pt = './model/microrts_GNNppo_' + date + '_10_worker.pt'
        record_path = './record/microrts_GNNppo_' + date + '_10.txt'
        torch.save(worker, path)
        torch.save(worker.state_dict(), path_pt)
        with open(record_path, "w") as f:
            f.write(str(rewards_record))
            f.write("\r\n")
            f.write(str(outcomes_record))

    if (time.time() - start_time) > 48 * 60 * 60:
        print('all game: ' + str(update))
        break
