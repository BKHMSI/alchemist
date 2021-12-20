import numpy as np
from tqdm import tqdm 

import torch
import torch.nn.functional as F

from dm_alchemy import symbolic_alchemy
from dm_alchemy.encode import chemistries_proto_conversion
from dm_alchemy.types import utils

from components.ep_memory import Episodic_Memory

def test(model, config, n_episodes = 100):

    chems = chemistries_proto_conversion.load_chemistries_and_items(
        'chemistries/perceptual_mapping_randomized_with_random_bottleneck/chemistries')

    device = config["device"]
    n_actions = config["task"]["n-actions"]
    n_potions = config["task"]["n-potions"]
    mem_dim = 1 + 5 + n_potions + 5 + 1
    episodic_memory = Episodic_Memory(1, model.mem_size, mem_dim)
    episode_rewards = []
    no_bottleneck_rewards = []


    for ep in tqdm(range(n_episodes)):

        chem, items = chems[ep]
        env = symbolic_alchemy.get_symbolic_alchemy_fixed(chemistry=chem, episode_items=items, see_chemistries={
            'input_chem': utils.ChemistrySeen(content=utils.ElementContent.GROUND_TRUTH)
        }, max_steps_per_trial=15)

        ht = torch.zeros(1, model.hidden_dim).float().to(device)
        ct = torch.zeros(1, model.hidden_dim).float().to(device)
        p_action = np.zeros((1, n_actions))
        p_reward = np.zeros((1, 1))

        trial_reward = 0
        episode_reward = 0
        episodic_memory.reset()

        timestep = env.reset()
        state = timestep.observation["symbolic_obs"]
        edges = timestep.observation["input_chem"][:12]
        n_edges = int(edges.sum())

        while not timestep.last():

            mem_mask = episodic_memory.generate_mask()
            states = np.array([state])
            model_states = convert_states(states)
            logit, _, (ht, ct) = model((
                torch.from_numpy(model_states).float().to(device),
                torch.from_numpy(p_action).float().to(device),
                torch.from_numpy(p_reward).float().to(device),
                torch.from_numpy(episodic_memory.memory).float().to(device),
                torch.from_numpy(mem_mask).float().to(device),
                (ht, ct)
            ))

            probs = F.softmax(logit, dim=-1)
            action = np.array([probs.argmax().item()])

            env_action = to_env_actions(states, action)

            timestep = env.step(env_action[0])
            done = timestep.last()
            s_prime = timestep.observation["symbolic_obs"]
        
            episode_reward += timestep.reward
            trial_reward += timestep.reward

            if env.is_new_trial():
                p_action = np.zeros((1, n_actions))
                p_reward = np.zeros((1, 1))
                state = s_prime 
            elif not done:
                stone_idx = (int(action)-1) // 7
                potion_color_idx = (int(action)-1) % 7

                if action == 0:
                    stone_idx = 0
                    potion_color_idx = 7                    
                
                stone_feats = state[stone_idx*5:(stone_idx+1)*5]
                stone_feats_p1 = s_prime[stone_idx*5:(stone_idx+1)*5]

                penalty = 0
                # if action doesn't have any effect on stone and action is not NoOp
                if all(state[:15]==s_prime[:15]) and int(action) != 0:
                    penalty = -0.2

                # choosing an empty or non-existent potion or using a cached stone
                elif all(state==s_prime) and int(action) != 0:
                    penalty = -1
                
                # choosing the same potion color consecutively 
                if int(action) == np.array(p_action).argmax() and int(action) % 7 != 0:
                    penalty += -1

                episodic_memory.push(np.array([
                    stone_idx,
                    *stone_feats,
                    *np.eye(n_potions)[potion_color_idx],
                    *stone_feats_p1,
                    timestep.reward + penalty,
                ]))

                reward = timestep.reward + penalty
                p_action = np.array([np.eye(n_actions)[int(action)]])
                p_reward = np.array([[reward]])
                state = s_prime 

        episode_rewards += [episode_reward]
        if n_edges == 12:
            no_bottleneck_rewards += [episode_reward]

        if (ep+1) % 100 == 0:
            print(f"> Reward Until Episode {ep+1}: {np.array(episode_rewards).mean()}")

    env.close()
    episode_rewards = np.array(episode_rewards)
    no_bottleneck_rewards = np.array(no_bottleneck_rewards)
    print(f"> Evaluated on {len(no_bottleneck_rewards)} Episodes: {no_bottleneck_rewards.mean()} ± {no_bottleneck_rewards.std() / np.sqrt(len(no_bottleneck_rewards))}")
    print(f"> Result: {episode_rewards.mean()} ± {episode_rewards.std() / np.sqrt(len(episode_rewards))}")
    return episode_rewards.mean()