import os
import pickle
import numpy as np

import torch
import torch.multiprocessing as mp

from dm_alchemy import symbolic_alchemy

def worker(env_seed, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging

    level_name = 'alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck'
    env = symbolic_alchemy.get_symbolic_alchemy_level(level_name, seed=env_seed, max_steps_per_trial=15)

    while True:
        cmd, action = worker_end.recv()
        if cmd == 'step':
            timestep = env.step(action)
            ob = timestep.observation["symbolic_obs"]
            reward = timestep.reward 
            done = timestep.last()
            worker_end.send((ob, reward, done))
        elif cmd == 'reset':
            timestep = env.reset()
            ob = timestep.observation["symbolic_obs"]
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_workers):
        self.nenvs = n_workers
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=((worker_id+1)*BASE_SEED, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

def compute_target(v_final, r_lst, mask_lst, gamma):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

def save_pickle(path, data):
    with open(os.path.join("tmp", path), 'wb') as f:
        pickle.dump(data, f)

def convert_states(states):
    n_workers = states.shape[0]
    stone_feats = states[:, :15]
    potions = np.zeros((n_workers, 6))
    for idx in range(15, 39, 2):
        potion_colors = np.array(list(map(int, np.round(states[:, idx]*3+3))))
        mask = states[:, idx+1] == 0
        potions[mask, potion_colors[mask]] += 1
    return np.concatenate([stone_feats, potions], axis=-1)

def to_env_actions(states, actions):
    stone_indices = (actions-1) // 7
    potion_color_indices = (actions-1) % 7

    potion_real_indices = np.ones_like(potion_color_indices) * -1
    for i, idx in enumerate(range(15, 39, 2)):
        potion_colors = np.array(list(map(int, np.round(states[:, idx]*3+3))))
        potion_mask = (potion_colors == potion_color_indices) * (states[:, idx+1] == 0)
        potion_real_indices[potion_mask] = i

    env_actions = stone_indices * 13 + 2 + potion_real_indices

    env_actions[potion_real_indices==-1] = 0

    env_actions[actions==0] = 0
    env_actions[actions==7] = 1
    env_actions[actions==14] = 14
    env_actions[actions==21] = 27

    return env_actions