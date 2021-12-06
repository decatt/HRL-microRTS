import torch
import random
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from env_wrapper import VecMonitor, VecPyTorch, MicroRTSStatsRecorder
import torch.optim as optim
from GNNAgent import Worker, Supervisor
import time
import torch.nn as nn
import collections
from dimension_reduction import get_selected_node_vector, get_round_imf
from numpy.random import choice


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
envs = MicroRTSVecEnv(
    num_envs=num_envs,
    max_steps=10000,
    render_theme=2,
    ai2s=ais,
    frame_skip=10,
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)

envs = MicroRTSStatsRecorder(envs, gamma)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)

worker = Worker(envs).to(device)
worker.load_state_dict(torch.load('./model/microrts_GNNppo_2021111319_10_worker.pt', map_location=device))
sup = Supervisor().to(device)
sup.load_state_dict(torch.load('./model/microrts_GNNppo_2021111707_10_sup.pt', map_location=device))

global_step = 0
start_time = time.time()
next_obs = envs.reset()
next_done = torch.zeros(num_envs).to(device)
num_updates = total_steps // batch_size # 128000000/8
starting_update = 1
gae_lambda = 0.95
sup_optimizer = optim.Adam(sup.parameters(), lr=learning_rate, eps=1e-5)

obs = torch.zeros((num_steps, num_envs, 7, 7, 27)).to(device)
actions = torch.zeros((num_steps, num_envs, 1)).to(device)
log_probs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)
d_masks = torch.zeros((num_steps, num_envs) + (4,)).to(device)

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
        sup_optimizer.param_groups[0]['lr'] = lrnow
    for step in range(0, num_steps):
        envs.render()
        global_step += 1 * num_envs
        graphs = torch.zeros((num_envs, 54))
        selected_units = torch.zeros((num_envs, size*size))
        round_imfs = torch.zeros((num_envs, 7, 7, 27))
        base_d = torch.zeros(num_envs)
        for i in range(num_envs):
            action_unit_mask = np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)[i]
            if sum(action_unit_mask) > 0:
                selected_unit = sample(action_unit_mask)
                graphs[i] = get_selected_node_vector(selected_unit, next_obs[i], size)
            else:
                selected_unit = 0
                graphs[i] = torch.zeros((54,))
            round_imf = get_round_imf(3, size, selected_unit, next_obs[i])
            round_imfs[i] = round_imf
            selected_units[i][selected_unit] = 1
            if selected_unit == 34 and (next_obs[i].permute((2, 0, 1))[17]*next_obs[i].permute((2, 0, 1))[11]).sum() > 4:
                base_d[i] = 1
        bef_obs = next_obs
        obs[step] = round_imfs
        dones[step] = next_done

        with torch.no_grad():
            values[step] = sup.get_value(obs[step]).flatten()

            action, _, _, mask = worker.get_action(graphs.to(device), num_envs, selected_units, envs=envs)

            action_d, log_prob, _, ms = sup.get_action(round_imfs.to(device), masks=mask.permute((1, 0))[106:110].permute((1, 0)))
            d_masks[step] = ms
            actions[step] = action_d.reshape(20, 1)
            log_probs[step] = log_prob
            action[2] = action_d
            action = action.T
            next_obs, rs, ds, infos = envs.step(action)
            for index in range(num_envs):
                step_rewards.append(rs[index].numpy()[0])
                if ds[index]:
                    if get_units_number(11, bef_obs, index) > get_units_number(12, bef_obs,index):
                        outcomes.append(1)
                    else:
                        outcomes.append(0)

        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
    if len(outcomes)>0:
        outcomes_record.append(sum(outcomes)/len(outcomes))
        rewards_record.append(sum(step_rewards)/len(step_rewards))
    # calculate advantage
    with torch.no_grad():
        last_value = sup.get_value(obs[-1].to(device)).reshape(1, -1)
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

        b_obs = obs.reshape((-1,) + (7, 7, 27))
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + (1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_invalid_action_masks = d_masks.reshape((-1, d_masks.shape[-1]))
    # sup learning
    target_sup = Supervisor().to(device)
    inds = np.arange(batch_size, )
    for i_epoch_pi in range(update_epochs):
        np.random.shuffle(inds)
        target_sup.load_state_dict(sup.state_dict())
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, new_log_prob, entropy, _ = sup.get_action(
                b_obs[minibatch_ind],
                action=b_actions.long()[minibatch_ind].T,
                masks=b_invalid_action_masks[minibatch_ind])
            ratio = (new_log_prob - b_log_probs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_log_probs[minibatch_ind] - new_log_prob).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = sup.get_value(b_obs[minibatch_ind]).view(-1)
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

            sup_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(sup.parameters(), max_grad_norm)
            sup_optimizer.step()

    if update % 64 == 0:
        date = time.strftime("%Y%m%d%H", time.localtime())
        path = './model/microrts_GNNppo_' + date + '_10_sup.pth'
        path_pt = './model/microrts_GNNppo_' + date + '_10_sup.pt'
        record_path = './record/microrts_GNNppo_' + date + '_10.txt'
        torch.save(sup, path)
        torch.save(sup.state_dict(), path_pt)
        with open(record_path, "w") as f:
            f.write(str(rewards_record))
            f.write("\r\n")
            f.write(str(outcomes_record))

    if (time.time() - start_time) > 48 * 60 * 60:
        print('all game: ' + str(update))
        break
