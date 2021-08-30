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
        z[0] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[16]).sum()
        z[0] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[17]).sum()
        z[0] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[18]).sum()
        z[0] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[19]).sum()
        z[0] = (state.permute((2, 0, 1))[11] * state.permute((2, 0, 1))[20]).sum()
        z[0] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[15]).sum()
        z[0] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[16]).sum()
        z[0] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[17]).sum()
        z[0] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[18]).sum()
        z[0] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[19]).sum()
        z[0] = (state.permute((2, 0, 1))[12] * state.permute((2, 0, 1))[20]).sum()
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


test_ai = False
use_gpu = True
seed = 0
num_envs = 20
torch_seeds = 0
gamma = 0.99
learning_rate = 2.5e-4
num_steps = 1024
total_steps = 25600000
n_minibatch = 4
anneal = True
c = 64
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
path = './model/microrts_fun_0830_16_worker.pth'
path_pt = './model/microrts_fun_0830_16_worker.pt'
path_manager = './model/microrts_fun_0830_16_manager.pth'
path_pt_manager = './model/microrts_fun_0830_16_manager.pt'
record_path = './record/microrts_ppo_0711_10.txt'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ais = []
for i in range(num_envs):
    ais.append(microrts_ai.workerRushAI)

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

manager = Manager().to(device)
worker = Worker(envs).to(device)

global_step = 0
start_time = time.time()
next_obs = envs.reset()
next_done = torch.zeros(num_envs).to(device)
num_updates = total_steps // batch_size
starting_update = 1
gae_lambda = 0.95
manager_optimizer = optim.Adam(manager.parameters(), lr=learning_rate, eps=1e-5)
worker_optimizer = optim.Adam(worker.parameters(), lr=learning_rate, eps=1e-5)
num_steps_manager = num_steps//c
obs_manager = torch.zeros((num_steps_manager, num_envs) + envs.observation_space.shape).to(device)
goals_manager = torch.zeros((num_steps_manager, num_envs) + (12,)).to(device)
rewards_manager = torch.zeros((num_steps_manager, num_envs)).to(device)
values_manager = torch.zeros((num_steps_manager, num_envs)).to(device)
log_probs_manager = torch.zeros((num_steps_manager, num_envs)).to(device)

manager_entropy = torch.zeros((num_envs, 12)).to(device)
obs = torch.zeros((num_steps, num_envs) + envs.observation_space.shape).to(device)
goals = torch.zeros((num_steps, num_envs) + (12,)).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.action_space.shape).to(device)
log_probs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)
invalid_action_masks = torch.zeros((num_steps, num_envs) + (envs.action_space.nvec.sum(),)).to(device)
embedded_obs = collections.deque(maxlen=2)
inner_rewards = 0

batch_size_manager = num_steps_manager*num_envs
minibatch_size_manager = int(batch_size_manager // n_minibatch)
num_updates_manager = total_steps // batch_size_manager

step_rewards = collections.deque(maxlen=5000)
outcomes = collections.deque(maxlen=500)
outcomes_record = []
rewards_record = []

for update in range(starting_update, num_updates + 1):
    if anneal:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        worker_optimizer.param_groups[0]['lr'] = lrnow
        manager_optimizer.param_groups[0]['lr'] = lrnow
    current_goal = torch.zeros(1, 12)
    for step in range(0, num_steps):
        envs.render()
        global_step += 1 * num_envs
        obs[step] = next_obs
        embedded_obs.append(embedding(obs[step], num_envs))
        dones[step] = next_done

        if step % c == 0 or step == 0:
            with torch.no_grad():
                current_goal, manager_log_probs, _ = manager.get_goal(obs[step], num_envs)
                goals[step] = current_goal
                log_probs_manager[step//c] = manager_log_probs
                values_manager[step//c] = manager.get_value(obs[step]).flatten()

        with torch.no_grad():
            values[step] = worker.get_value(obs[step], current_goal).flatten()
            action, log_prob, _, invalid_action_masks[step] = worker.get_action(obs[step], current_goal, num_envs,
                                                                                envs=envs)
            actions[step] = action.T
            log_probs[step] = log_prob
            next_obs, rs, ds, infos = envs.step(action.T)
            for index in range(num_envs):
                step_rewards.append(rs[index].numpy()[0])
                if ds[index]:
                    if get_units_number(11, obs[step], index) > get_units_number(12, obs[step], index):
                        outcomes.append(1)
                    else:
                        outcomes.append(0)

        if len(embedded_obs) == 2:
            inner_rewards = get_inner_reward(current_goal, embedded_obs[0], embedded_obs[1], num_envs)
        else:
            inner_rewards = torch.zeros((num_envs,))

        rewards[step], next_done = rs.view(-1) + inner_rewards, torch.Tensor(ds).to(device)
        rewards_manager[step // c] = rewards_manager[step // c] + rewards[step]
    if len(outcomes) > 0:
        outcomes_record.append(sum(outcomes) / len(outcomes))
        rewards_record.append(sum(step_rewards) / len(step_rewards))

    with torch.no_grad():
        last_value = worker.get_value(next_obs.to(device), goals[-1].to(device)).reshape(1, -1)
        if gae:
            advantages = torch.zeros_like(rewards).to(device)
            advantages_manager = torch.zeros_like(rewards_manager).to(device)
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
                advantages_manager[t//c] = advantages_manager[t//c] + advantages[t]
            returns = advantages + values
            returns_manager = advantages_manager + values_manager
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

        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_obs_manager = obs_manager.reshape((-1,) + envs.observation_space.shape)
        b_goals = goals.reshape((-1,) + (12,))
        b_log_probs = log_probs.reshape(-1)
        b_log_probs_manager = log_probs_manager.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_values_manager = values_manager.reshape(-1)
        b_invalid_action_masks = invalid_action_masks.reshape((-1, invalid_action_masks.shape[-1]))
        b_advantages_manager = advantages_manager.reshape(-1)
        b_returns_manager = returns_manager.reshape(-1)
    # manager learning
    if manager_learn:
        target_manager = Manager().to(device)
        inds_m = np.arange(batch_size_manager, )
        for i_epoch_pi in range(update_epochs):
            np.random.shuffle(inds_m)
            target_manager.load_state_dict(manager.state_dict())
            for start in range(0, batch_size_manager, minibatch_size_manager):
                end = start + minibatch_size_manager
                minibatch_ind_m = inds_m[start:end]
                mb_advantages_manager = b_advantages_manager[minibatch_ind_m]
                if norm_adv:
                    mb_advantages_manager = (mb_advantages_manager - mb_advantages_manager.mean()) / (mb_advantages_manager.std() + 1e-8)
                _, new_log_prob_manager, entropy_manager = manager.get_goal(b_obs_manager[minibatch_ind_m], minibatch_size_manager)
                ratio = (new_log_prob_manager - b_log_probs_manager[minibatch_ind_m]).exp()
                approx_kl = (b_log_probs_manager[minibatch_ind_m] - new_log_prob_manager).mean()
                pg_loss1 = -mb_advantages_manager * ratio
                pg_loss2 = -mb_advantages_manager * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy_manager.mean()

                new_values = manager.get_value(b_obs_manager[minibatch_ind_m]).view(-1)
                if clip_vloss:
                    v_loss_unclipped = ((new_values - b_returns_manager[minibatch_ind_m]) ** 2)
                    v_clipped = b_values_manager[minibatch_ind_m] + torch.clamp(new_values - b_values[minibatch_ind_m], -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - b_returns_manager[minibatch_ind_m]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns_manager[minibatch_ind_m]) ** 2)

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                manager_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(worker.parameters(), max_grad_norm)
                manager_optimizer.step()

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
                b_goals[minibatch_ind],
                num_envs=num_envs,
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
            new_values = worker.get_value(b_obs[minibatch_ind], b_goals[minibatch_ind]).view(-1)
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

    if update % 100 == 0:
        torch.save(worker, path)
        torch.save(worker.state_dict(), path_pt)
        torch.save(manager, path_manager)
        torch.save(manager.state_dict(), path_pt_manager)

    if (time.time() - start_time) > 20 * 60 * 60:
        print('all game: ' + str(update))
        break

with open(record_path, "w") as f:
    f.write(str(rewards_record))
    f.write("\r\n")
    f.write(str(outcomes_record))


fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.plot(np.array(range(len(rewards_record))), np.array(rewards_record), color='red')
ax2.plot(np.array(range(len(outcomes_record))), np.array(outcomes_record), color='red')
fig.canvas.draw()
plt.show()
