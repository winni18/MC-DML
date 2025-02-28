import os
import json
import argparse
import time
import jericho
import random
import numpy as np
from collections import defaultdict

from src.mcts import MCTSAgent
from src.env import JerichoEnv
import src.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Game
    parser.add_argument('--rom_path', default='envs/jericho-game-suite/', type=str)
    parser.add_argument('--game_name', default='balances', type=str)
    parser.add_argument('--data_path', default='data/GAME', type=str)
    parser.add_argument('--env_step_limit', default=100000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    # MCTS
    parser.add_argument('--uct_type', default='MC-DML', type=str)
    parser.add_argument('--exploration_constant', default=50, type=int)
    parser.add_argument('--max_episode_len', default=100, type=int)
    parser.add_argument('--initial_max_depth', default=15, type=int)
    parser.add_argument('--allowed_max_depth', default=20, type=int)
    parser.add_argument('--simulation_per_act', default=50, type=int) 
    parser.add_argument('--discount_factor', default=0.95, type=float)

    # LLM
    parser.add_argument('--llm_model', default='gpt-3.5-turbo', type=str)
    parser.add_argument('--llm_temperature', default=0, type=int)
    parser.add_argument('--max_memory', default=3, type=int)
    parser.add_argument('--softmax_temperature', default=5, type=int)
    return parser.parse_args()


def main():    
    args = parse_args()
    print(args)

    args.rom_path = args.rom_path + utils.game_file(args.game_name)
    data_path = args.data_path.replace('GAME', args.game_name)

    if args.seed is None:
        args.seed = random.randint(0,100)
    np.random.seed(args.seed)

    log_dir = data_path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    valid_action_dict = {}
    
    assert jericho.__version__.startswith('3'), "This code is designed to be run with Jericho version >= 3.0.0."
    env = JerichoEnv(rom_path=args.rom_path, 
                     seed=args.seed, 
                     step_limit=args.env_step_limit, 
                     get_valid = True,
                     cache = valid_action_dict
                     )

    
    ob, info = env.reset()
    visited_transitions = []  
    done = False
    cum_reward = info['score']
    step = 0
    prev_action = '<START>'

    log_file_path = log_dir + 'mcts_log_d%02d_d%02d_s%d_e%d_%02d.txt'\
               % (args.initial_max_depth, args.allowed_max_depth, args.simulation_per_act, args.exploration_constant, args.seed)    
    replay_buffer_filename = log_dir + 'mcts_replay_d%02d_%02d.txt' % (args.initial_max_depth, args.seed)

    
    start = time.time()

    agent = MCTSAgent(args, env,
                      visited_transitions=visited_transitions,
                      replay_file=None)

    with open(log_file_path, 'w', encoding='utf-8') as data, open(replay_buffer_filename, 'w', encoding='utf-8') as replay_buffer_file:
        agent.replay_file = replay_buffer_file


        for cur_depth in range(args.max_episode_len):

            root_node, action, visited_transitions = agent.search(ob, info, cur_depth)
            
            step_str = f'[STEP] {step}\n'
            prev_action_str = f'[PREV_ACTION] {prev_action}\n'
            state_str = (f'[OBS] {ob}\n'
                            f'[LOOK] {info["look"]}\n'
                            f'[INV] {info["inv"]}\n')
            valid_actions_str = ''.join(f'[VALID_ACTION] {valid}\n' for valid in info['valid'])
            action_str = f'[ACTION] {action}\n'
            full_output = f"{step_str}{state_str}{valid_actions_str}{action_str}{prev_action_str}"
            data.write(full_output)

            ob, reward, done, info = env.step(action)

            cum_reward += reward
            score = info['score']
            step += 1

            data.write(f"Reward: {reward}, Cum_reward: {score}\n")
            for action_node, prob in zip(root_node.children, root_node.action_probs[-1]):
                data.write('%s Q: %f Count: %d Prob: %f  \n' % (action_node.action, action_node.Q, action_node.N, prob))

            data.flush()

            print('STEP: %s' % step)
            print(root_node.state)
            print('BEST_ACTION: ', action)
            print('Valid actions:', [action.action for action in root_node.children])
            print('Q-values', [action.Q for action in root_node.children])
            print('Maximum Q', [0 if len(action.Rs) == 0 else max(action.Rs) for action in root_node.children])
            print('Count of actions', [action.N for action in root_node.children])
            print('LLM Probs:', [prob for prob in root_node.action_probs[-1]])
            print('Reward: %s, CUM_Reward: %s' % (reward, score))
            print(ob + info['look'] + info['inv'])

            prev_action = action
            
            if done:
                break

        total_time = time.time() - start
        print(f"TOTAL TIME: {total_time:.2f} seconds")

        env.close()

if __name__ == "__main__":
    main()
