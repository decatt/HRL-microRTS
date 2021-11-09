import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from env_wrapper import VecMonitor, VecPyTorch, MicroRTSStatsRecorder
import torch.optim as optim
from feudal_agent import Manager, Worker, ManagerScript
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
import time


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
        vector_b = v_zs[i_env] - v_zs_[i_env]
        num = torch.mm(vector_a.view((1, 12)), vector_b.view((12, 1)))
        denom = torch.norm(vector_a) * torch.norm(vector_b)
        if denom == 0:
            res[i_env] = 0
        else:
            cos = num / denom
            sim = cos
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
total_steps = 128000000
n_minibatch = 4
anneal = True
c = 16
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
path = './model/microrts_fun_'+date+'_10_worker.pth'
path_pt = './model/microrts_fun_'+date+'_10_worker.pt'
path_manager = './model/microrts_fun_'+date+'_10_manager.pth'
path_pt_manager = './model/microrts_fun_'+date+'_10_manager.pt'
record_path = './record/microrts_fun_'+date+'_10.txt'
path_hp = './model/microrts_fun_'+date+'_10_worker_hp.pth'
path_pt_hp = './model/microrts_fun_'+date+'_10_worker_hp.pt'
path_manager_hp = './model/microrts_fun_'+date+'_10_manager_hp.pth'
path_pt_manager_hp = './model/microrts_fun_'+date+'_10_manager_hp.pt'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ais = []
for i in range(num_envs):
    ais.append(microrts_ai.coacAI)

init_seeds()
envs = MicroRTSVecEnv(
    num_envs=num_envs,
    max_steps=10000,
    render_theme=2,
    ai2s=ais,
    frame_skip=10,
    map_path="maps/10x10/basesWorkers10x10.xml",
    reward_weight=np.array([10.0, 1.0, 0, 0.2, 1.0, 0])
)
# build worker 1 090604
# resource 1 090611
# attack 1 090619
# build barracks units 090701


envs = MicroRTSStatsRecorder(envs, gamma)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)

manager = ManagerScript(num_envs)
worker = Worker(envs).to(device)

global_step = 0
start_time = time.time()
next_obs = envs.reset()
next_done = torch.zeros(num_envs).to(device)
num_updates = total_steps // batch_size
starting_update = 1
gae_lambda = 0.95
worker_optimizer = optim.Adam(worker.parameters(), lr=learning_rate, eps=1e-5)


obs = torch.zeros((num_steps, num_envs) + envs.observation_space.shape).to(device)
goals = torch.zeros((num_steps, num_envs) + (12,)).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.action_space.shape).to(device)
log_probs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)
invalid_action_masks = torch.zeros((num_steps, num_envs) + (envs.action_space.nvec.sum(),)).to(device)
embedded_obs = collections.deque(maxlen=2)
inner_rewards = torch.zeros((num_envs,))
inner_rewards_weight = 10

step_rewards = collections.deque(maxlen=5000)
outcomes = collections.deque(maxlen=500)
outcomes_record = []
rewards_record = []

hp_outcome = 0.0

for update in range(starting_update, num_updates + 1):
    if anneal:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        worker_optimizer.param_groups[0]['lr'] = lrnow
    current_goal = torch.zeros((num_envs, 12)).to(device)
    for step in range(0, num_steps):
        envs.render()
        global_step += 1 * num_envs
        obs[step] = next_obs
        zt = embedding(obs[step], num_envs)
        embedded_obs.append(zt)
        dones[step] = next_done
        if 0 == (current_goal != zt).sum():
            current_goal = manager.get_goal(zt, num_envs)
            goals[step] = current_goal
        if step % c == 0 or step == 0:
            current_goal = manager.get_goal(zt, num_envs)
            goals[step] = current_goal
        else:
            for index in range(num_envs):
                if ds[index]:
                    current_goal = manager.get_goal(zt, num_envs)
                    goals[step] = current_goal
        with torch.no_grad():
            current_goal = torch.round(current_goal)
            values[step] = worker.get_value(obs[step], current_goal).flatten()

            action, log_prob, _, invalid_action_masks[step] = worker.get_action(obs[step], current_goal, num_envs,
                                                                                envs=envs)
            actions[step] = action.T
            log_probs[step] = log_prob
            next_obs, rs, ds, infos = envs.step(action.T)

        if len(embedded_obs) == 2 and (0 != (embedded_obs[0]!=embedded_obs[1]).sum()):
            inner_rewards = get_inner_reward(current_goal, embedded_obs[0], embedded_obs[1], num_envs)
        else:
            inner_rewards = torch.zeros((num_envs,))
        rewards[step], next_done = rs.view(-1) + inner_rewards_weight*inner_rewards, torch.Tensor(ds).to(device)
    if len(outcomes) > 0:
        average_outcomes = sum(outcomes) / len(outcomes)
        if average_outcomes > hp_outcome:
            torch.save(worker, path_hp)
            torch.save(worker.state_dict(), path_pt_hp)
            torch.save(manager, path_manager_hp)
            torch.save(manager.state_dict(), path_pt_manager_hp)
            hp_outcome = average_outcomes
            print(hp_outcome)
        outcomes_record.append(sum(outcomes) / len(outcomes))
        rewards_record.append(sum(step_rewards) / len(step_rewards))

    with torch.no_grad():
        last_value = worker.get_value(next_obs.to(device), goals[-1].to(device)).reshape(1, -1)
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
            for t in reversed(range(
                    num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_return = last_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * next_non_terminal * next_return
                advantages = returns - values

        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_goals = goals.reshape((-1,) + (12,))
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_invalid_action_masks = invalid_action_masks.reshape((-1, invalid_action_masks.shape[-1]))
    # manager learning
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

    if update % 64 == 0:
        date = time.strftime("%Y%m%d%H", time.localtime())
        path = './model/microrts_fun_' + date + '_10_worker.pth'
        path_pt = './model/microrts_fun_' + date + '_10_worker.pt'
        record_path = './record/microrts_fun_' + date + '_10.txt'
        torch.save(worker, path)
        torch.save(worker.state_dict(), path_pt)
        with open(record_path, "w") as f:
            f.write(str(rewards_record))
            f.write("\r\n")
            f.write(str(outcomes_record))

    if (time.time() - start_time) > 72 * 60 * 60:
        print('all game: ' + str(update))
        break


fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.plot(np.array(range(len(rewards_record))), np.array(rewards_record), color='red')
ax2.plot(np.array(range(len(outcomes_record))), np.array(outcomes_record), color='red')
fig.canvas.draw()
plt.show()

