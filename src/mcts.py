import json
import numpy as np
from tqdm import tqdm
from . import utils
from collections import defaultdict, deque

from typing import List, Dict, Any
from .llm import LLMAgent
from .openai_helpers import chat_completion_with_retries, truncate_text


class StateNode:
    def __init__(self, reward=0, done=False):
        self.ob = None
        self.look = None
        self.inv = None
        self.state = None
        self.prev_state = None
        self.prev_action = None
        self.id = None
        self.valid_actions = None
        self.action_probs = deque(maxlen=3)

        self.N = 0
        self.children = []
        self.reward = reward
        self.score = 0
        self.done = done


class ActionNode:
    def __init__(self, action):
        self.action = action
        self.N = 0
        self.Q = 0
        self.Rs = []
        self.children = []
        self.children_text = []
               
               
class MCTSAgent:
    
    def __init__(self, args, env, visited_transitions=None, replay_file=None):
        
        self.env = env
        self.llm = LLMAgent(args)

        self.exploration_constant = args.exploration_constant
        self.initial_max_depth = args.initial_max_depth
        self.allowed_max_depth = args.allowed_max_depth
        self.simulation_per_act = args.simulation_per_act
        self.discount_factor = args.discount_factor
        self.max_memory = args.max_memory
        
        self.visited_transitions = visited_transitions       
        self.replay_file = replay_file
        
        self.trajectory = []
        self.memory_per_loop = 0 
        self.memory = deque(maxlen=self.max_memory)
        

    def build_state(self, ob, info, reward=0, done=False, prev_state='<s>', prev_action='<s>'):
        state = StateNode()
        
        state.ob = ob
        state.look = info['look']
        state.inv = info['inv']
        state.state = ob + info['look'] + info['inv']
        state.reward = reward
        state.score = info['score']
        state.prev_action = prev_action
        state.prev_state = prev_state
        
        if ob == info['look']:
            state.state = ob + info['inv']
            state.id = ob + info['inv'] + str(reward) + str(info['score']) + prev_action
        else:
            state.state = ob + info['look'] + info['inv']
            state.id = ob + info['look'] + info['inv'] + str(reward) + str(info['score']) + prev_action

        state.valid_actions = info['valid']
        state.done = done

        for valid_action in info['valid']:
            state.children.append(ActionNode(valid_action))

        return state
   

    def search(self, ob, info, cur_depth):  

        current_max_depth = self.initial_max_depth
        
        while current_max_depth <= self.allowed_max_depth:
            self.root = self.build_state(ob, info)
            self.memory_per_loop = 0
            self.memory = deque(maxlen=3)
            
            for _ in tqdm(range(self.simulation_per_act * len(self.root.children))):
                copy_env = self.env.copy()
                self.trajectory = []
                self.simulate(self.root, copy_env, 0, current_max_depth)
                copy_env.close()
            
            best_action_node = self.greedy_action_node(self.root, 0)

            if len(best_action_node.Rs) != 0 and max(best_action_node.Rs) != 0:
                return self.root, best_action_node.action, self.visited_transitions
            else:
                current_max_depth += 20

        return self.root, best_action_node.action, self.visited_transitions



    def write_buffer(self, state_node, best_action_node, ob, reward, done, info):
        
        def clean_string(input_str):
            return " ".join(input_str.split('\n')).strip()

        obs = '[OBS] ' + clean_string(state_node.ob)
        look = '[LOOK] ' + clean_string(state_node.look)
        inven = '[INV] ' + clean_string(state_node.inv)
        sign = '0' if int(state_node.score) >= 0 else '1'
        score_string = '{0:09b}'.format(abs(int(state_node.score)))
        score = '[SCORE] %s%s ' % (sign, score_string)

        prev_action = '[PREV_ACTION] ' + clean_string(state_node.prev_action)
        action = '[ACTION] ' + clean_string(best_action_node.action)
        valid_actions = '[VALID_ACTION] ' + ' [VALID_ACTION] '.join(state_node.valid_actions)
        
        next_sign = '0' if int(state_node.score + reward) >= 0 else '1'
        next_score_string = '{0:09b}'.format(abs(int(state_node.score  + reward)))

        reward = '[REWARD] %d ' % reward
        next_score = '[NEXT_SCORE] %s%s ' % (next_sign, next_score_string)
        
        done = '[DONE] %d ' % done
        next_obs = '[NEXT_OBS] ' + clean_string(ob)
        next_look = '[NEXT_LOOK] ' + clean_string(info['look'])
        next_inven = '[NEXT_INV] ' + clean_string(info['inv'])

        transition = [obs, look, inven, score, prev_action, action, valid_actions, reward, done, next_obs, next_look, next_inven, next_score]
        transition_str = " ".join(transition) + '\n'
        
        if self.replay_file is not None:
            self.replay_file.write(transition_str)
        

    def simulate(self, state_node, copy_env, depth, max_depth):
        
        if state_node.done or depth == max_depth or (state_node.look == 'unknown' and state_node.inv == 'unknown'):
            
            if state_node.done and self.memory_per_loop < self.max_memory:
                reflection = self.llm.get_traj_reflection(self.trajectory)
                self.memory.append(reflection)
                self.memory_per_loop += 1 
            return 0
        
        best_action_node = self.greedy_action_node(state_node, self.exploration_constant)

        rollout_next = False

        if state_node.ob == state_node.look:
            prev_state = state_node.ob + state_node.inv
        else:
            prev_state = state_node.ob + state_node.look + state_node.inv

        ob, reward, done, info = copy_env.step(best_action_node.action)
        next_state_text = ob + info['look'] + info['inv']
        
        if "*** you have died ***" in next_state_text:
            reward = -10

        self.write_buffer(state_node, best_action_node, ob, reward, done, info)
        self.trajectory.append({'state': state_node.state, 'action': best_action_node.action})

        if next_state_text in best_action_node.children_text:
            index = best_action_node.children_text.index(next_state_text)
            next_state_node = best_action_node.children[index]

            if next_state_node.N == 0:
                rollout_next = True
            next_state_node.N += 1

        else:
            next_state_node = self.build_state(ob, info, reward, done, prev_state=prev_state, prev_action=best_action_node.action)
            best_action_node.children.append(next_state_node)
            best_action_node.children_text.append(next_state_node.state)
            rollout_next = True

        if rollout_next:
            R = reward + self.discount_factor * self.rollout(next_state_node, copy_env, depth+1, max_depth)
        else:
            R = reward + self.discount_factor * self.simulate(next_state_node, copy_env, depth+1, max_depth)

        state_node.N += 1
        best_action_node.N += 1

        best_action_node.Rs.append(R)
        best_action_node.Q = np.sum(np.array(best_action_node.Rs) * utils.softmax(best_action_node.Rs, T=10))

        return R


    def greedy_action_node(self, state_node, exploration_constant):

        if len(self.memory) == 0:
            if len(state_node.action_probs) == 0:
                _, child_score_list = self.llm.get_action_probs(state_node, list(self.memory))
                state_node.action_probs.append(child_score_list)
            else:
                child_score_list = state_node.action_probs[-1]
        
        else:
            if len(state_node.action_probs) <= len(self.memory):
                _, child_score_list = self.llm.get_action_probs(state_node, list(self.memory))
                state_node.action_probs.append(child_score_list)
            else:
                child_score_list = state_node.action_probs[-1]
        
        best_value = -float('inf')
        best_children = []
        
        for i, child in enumerate(state_node.children):
            child_llm_prob = child_score_list[i]  
            ucb_value = child.Q + exploration_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child_llm_prob

            if np.isclose(ucb_value, best_value):
                best_children.append(child)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [child]

        return np.random.choice(best_children)




    def rollout(self, state_node, copy_env, depth, max_depth):
        
        if state_node.done or depth == max_depth or (state_node.look == 'unknown' and state_node.inv == 'unknown'):
            if state_node.done and self.memory_per_loop < 3:
                reflection = self.llm.get_traj_reflection(self.trajectory)
                self.memory.append(reflection)
                self.memory_per_loop += 1 
            return 0
        

        action_node = np.random.choice(state_node.children, 1)[0]
        action = action_node.action
        
        if state_node.ob == state_node.look:
            prev_state = state_node.ob  + state_node.inv
        else:
            prev_state = state_node.ob  + state_node.look + state_node.inv
            
        ob, reward, done, info = copy_env.step(action)
        
        next_state_text = ob + info['look'] + info['inv']
        
        if "*** you have died ***" in next_state_text:
            reward = -10

        self.write_buffer(state_node, action_node, ob, reward, done, info)
        self.trajectory.append({'state': state_node.state, 'action': action_node.action})

        if next_state_text in action_node.children_text:
            index = action_node.children_text.index(next_state_text)
            next_state_node = action_node.children[index]
        else:
            next_state_node = self.build_state(ob, info, reward, done, prev_state=prev_state, prev_action=action)
            action_node.children.append(next_state_node)
            action_node.children_text.append(next_state_node.state)

        return reward + self.discount_factor * self.rollout(next_state_node, copy_env, depth+1, max_depth)


