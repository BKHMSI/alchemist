import os 
import yaml
import argparse
import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from components.ep_memory import Episodic_Memory
from model import A2C_EPN
from test import test 
from utils import ParallelEnv, to_env_actions, convert_states, compute_target

BASE_SEED = 42

def main(config):
    device = config["device"]

    learning_rate = config["agent"]["lr"]
    gamma = config["agent"]["gamma"]
    val_coeff = config["agent"]["value-loss-weight"]
    entropy_coeff_base = config["agent"]["entropy-weight"]
    grad_clip_norm = config["agent"]["grad-clip-norm"]
    n_actions = config["task"]["n-actions"]
    n_potions = config["task"]["n-potions"]
    n_workers = config["agent"]["n-workers"]
    n_episodes = config["task"]["n-episodes"]
    mem_size = config["agent"]["dict-len"]
    mem_dim = 1 + 5 + n_potions + 5 + 1
    save_interval = config["save-interval"]
    update_interval = config["agent"]["n-step-update"]
    steps_per_trial = 15

    save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")
    writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"], config["run-title"]))

    model = A2C_EPN(config["agent"], n_actions)
    model.to(device)
    model.train()

    if config["resume"] or config["test"]:
        filepath = config["load-path"]
        print(f"> Loading Checkpoint {filepath}")
        model_data = torch.load(filepath, map_location=torch.device(config["device"]))
        model.load_state_dict(model_data["state_dict"])


    if config["test"]:
        print(f"> Evaluating Episodes")
        model.eval()
        test(model, config, n_episodes=1000)
        exit()
    
    envs = ParallelEnv(n_workers)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    lambda_lr = lambda x: max(0.1, float(100_000 - x) / float(100_000))
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    entropy_scheduler = lambda x: max(0.01, float(100_000 - x) / float(100_000))

    step_idx = 0
    states = envs.reset()

    episodic_memory = Episodic_Memory(n_workers, mem_size, mem_dim)
    p_action = np.zeros((n_workers, n_actions))
    p_reward = np.zeros((n_workers, 1))
    done = np.array([True]*n_workers)  
    episode_rewards = np.zeros(n_workers)
    steps_this_trial = 1
    episode_num = 0
    is_optim_step = False 
    update_counter = 0
    total_rewards = []
    entropy_coeff = entropy_coeff_base
    progress = tqdm(np.arange(n_episodes))

    while episode_num < n_episodes:
        r_lst, mask_lst = list(), list()
        values, log_probs, entropies = list(), list(), list()

        if all(done):
            ht = torch.zeros((n_workers, model.hidden_dim)).float().to(device)    
            ct = torch.zeros((n_workers, model.hidden_dim)).float().to(device)  
        else:
            ht, ct = ht.detach(), ct.detach()

        for _ in range(update_interval):
            
            mem_mask = episodic_memory.generate_mask()
            model_states = convert_states(states)
            logit, value, (ht, ct) = model((
                torch.from_numpy(model_states).float().to(device),
                torch.from_numpy(p_action).float().to(device),
                torch.from_numpy(p_reward).float().to(device),
                torch.from_numpy(episodic_memory.memory).float().to(device),
                torch.from_numpy(mem_mask).float().to(device),
                (ht, ct)
            ))

            probs = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=True)
            actions = Categorical(probs).sample().detach()
            log_prob = log_prob.gather(1, actions.unsqueeze(1))
            actions = actions.cpu().numpy()

            env_actions = to_env_actions(states, actions)
            s_prime, rewards, done = envs.step(env_actions)

            episode_rewards += rewards

            ######### New Trial #########
            if steps_this_trial == steps_per_trial:
                steps_this_trial = 1
                p_action = np.zeros((n_workers, n_actions))
                p_reward = np.zeros((n_workers, 1))
                states = s_prime
            elif not all(done):
            
                stone_indices = (actions-1) // 7
                potion_color_indices = (actions-1) % 7

                stone_indices[actions==0] = 0
                potion_color_indices[actions==0] = 7

                stone_feats = np.zeros((n_workers, 5))
                stone_feats_p1 = np.zeros((n_workers, 5))
                for worker_idx, stone_idx in enumerate(stone_indices):
                    stone_feats[worker_idx] = states[worker_idx, stone_idx*5:(stone_idx+1)*5]
                    stone_feats_p1[worker_idx] = s_prime[worker_idx, stone_idx*5:(stone_idx+1)*5]
                potion_color_onehot = np.eye(n_potions)[potion_color_indices]


                penalty = np.zeros(n_workers)

                # if action doesn't have any effect on stone and action is not NoOp
                pen_mask = (states[:, :15]==s_prime[:, :15]).all(-1) * (actions != 0)
                penalty[pen_mask] = -0.2

                # choosing an empty or non-existent potion or using a cached stone
                pen_mask = (states==s_prime).all(-1) * (actions != 0)
                penalty[pen_mask] = -1

                # choosing the same potion color consecutively 
                pen_mask = (actions % 7 != 0) * (actions == p_action.argmax(-1))
                penalty[pen_mask] += -1

                rewards += penalty

                memories = np.concatenate([
                    stone_indices[:, np.newaxis], 
                    stone_feats, 
                    potion_color_onehot, 
                    stone_feats_p1,
                    rewards[:, np.newaxis]
                ], axis=-1)

                episodic_memory.push(memories)

                p_reward = rewards[:, np.newaxis]
                p_action = np.eye(n_actions)[actions]
                states = s_prime

                steps_this_trial += 1

            values.append(value)
            entropies.append(entropy)
            log_probs.append(log_prob)
            r_lst.append(rewards)
            mask_lst.append(1 - done)
            step_idx += 1

            if all(done):
                states = envs.reset()
                p_action = np.zeros((n_workers, n_actions))
                p_reward = np.zeros((n_workers, 1))
  
                episodic_memory.reset()
                episode_num += 1

                if is_optim_step:
                    lr_scheduler.step()
                    last_lr = lr_scheduler.get_last_lr()[0]
                    entropy_coeff = entropy_coeff_base * entropy_scheduler(episode_num)
                else:
                    last_lr = learning_rate

                total_rewards += [episode_rewards.mean()]
                avg_reward_100 = np.array(total_rewards[-100:]).mean()
                writer.add_scalar("perf/reward_t", episode_rewards.mean(), episode_num)
                writer.add_scalar("perf/avg_reward_100", avg_reward_100, episode_num)
                writer.add_scalar("perf/lr", last_lr, episode_num)
                writer.add_scalar("perf/entropy_coeff", entropy_coeff, episode_num)
                episode_rewards = np.zeros(n_workers)

                progress.update()

                if episode_num % save_interval == 0:
                    test_reward_100 = test(model, config)
                    writer.add_scalar("perf/test_reward_100", test_reward_100, episode_num)
                    torch.save({
                        "test_reward_100": test_reward_100,
                        "state_dict": model.state_dict(),
                        "avg_reward_100": avg_reward_100,
                    }, save_path.format(epi=episode_num) + ".pt")

        mem_mask = episodic_memory.generate_mask()
        model_states = convert_states(s_prime)
        _, value, _ = model((
            torch.from_numpy(model_states).float().to(device),
            torch.from_numpy(np.eye(n_actions)[actions]).float().to(device),
            torch.from_numpy(rewards[:, np.newaxis]).float().to(device),
            torch.from_numpy(episodic_memory.memory).float().to(device),
            torch.from_numpy(mem_mask).float().to(device),
            (ht, ct)
        ))

        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        td_target = compute_target(value.detach().cpu().numpy(), r_lst, mask_lst, gamma)
        td_target = td_target.to(device)

        td_target_vec = td_target.reshape(-1)
        advantage = td_target_vec - values.reshape(-1)
        action_loss = -(log_probs.reshape(-1) * advantage.detach()).mean()
        value_loss = F.smooth_l1_loss(values.reshape(-1), td_target_vec) # or advantage.pow(2).mean()
        entropy = entropies.reshape(-1).mean()

        loss = value_loss * val_coeff + action_loss - entropy * entropy_coeff

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        is_optim_step = True 

        writer.add_scalar("losses/total_loss", loss.item(), update_counter)
        writer.add_scalar("losses/value_loss", value_loss.item(), update_counter)
        writer.add_scalar("losses/action_loss", action_loss.item(), update_counter)
        writer.add_scalar("losses/entropy", entropy.item(), update_counter)
        update_counter += 1

    envs.close()

if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    exp_path = os.path.join(config["save-path"], config["run-title"])
    if not os.path.isdir(exp_path): 
        os.mkdir(exp_path)
    
    out_path = os.path.join(exp_path, os.path.basename(args.config))
    with open(out_path, 'w') as fout:
        yaml.dump(config, fout)

    print("="*50)
    print(f"Running {config['run-title']}")
    print("="*50)

    main(config)