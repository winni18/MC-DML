import json
from typing import List, Dict
from collections import deque

from .openai_helpers import chat_completion_with_retries, truncate_text
from .utils import softmax


class LLMAgent:
    """
    LLM Agent for selecting actions in a text-based adventure game.
    """
    def __init__(self, args):
        self.model = args.llm_model
        self.max_memory = args.max_memory
        self.llm_temperature = args.llm_temperature
        self.softmax_temperature = args.softmax_temperature


    def _format_state(self, state_node):
        return f"PREV_STATE: {state_node.prev_state}\nACTION: {state_node.prev_action}\nCURRENT_STATE: {state_node.state}"


    def get_probs_prompts(self, state_node, memory):

        formatted_state = self._format_state(state_node)
        actions_str = [f"{i}: {a}" for i, a in enumerate(state_node.valid_actions)]
        formatted_actions = "\n".join(actions_str)

        sys_prompt = """You are a player in a text-based adventure game. Your task is to evaluate and select actions that are promising based on the given game state."""

        if memory:
            user_prompt = f"""Your memory of playing this game previously is: {memory}
            You are now facing the following state in the game:{formatted_state}
            Considering the current state and previous memories, please select the action most worth exploring from the following list: {formatted_actions}
            Respond by providing the index of the action only. Your response should be a single integer, without any extra formatting, spaces, punctuation, or text."""
        else:
            user_prompt = f"""You are now facing the following state in the game: {formatted_state}
            Considering the current state, please select the most promising action from the following list: {formatted_actions}
            Respond by providing the index of the action only. Your response should be a single integer, without any extra formatting, spaces, punctuation, or text."""       
        return sys_prompt, user_prompt
    

    def get_reflection_prompts(self, trajectory):
        text_trajectory = "\n".join(
            f"STEP {i}, STATE: {step['state']}, ACTION: {step['action']}"
            for i, step in enumerate(trajectory)
        )
        text_trajectory = truncate_text(text_trajectory, 5000)

        sys_prompt = """You will receive a log of unsuccessful gameplay from a text-based adventure game. Please identify the reasons for this game failure and provide a short suggestion for improving the game strategy next time. Do not summarize the gameplay trajectory; respond with your suggestion in a single sentence. For instance:  'Remember to light a lamp before entering dark areas to avoid being eaten by a grue. '"""
        user_prompt = f"""GAMEPLAY TRAJECTORY: \n{text_trajectory}"""
        return sys_prompt, user_prompt
    
    
    def get_action_probs(self, state_node, memory):
        sys_prompt, user_prompt = self.get_probs_prompts(state_node, memory)

        valid_labels = [str(i) for i in range(len(state_node.valid_actions))]

        res = chat_completion_with_retries(
            model=self.model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=2,
            temperature=self.llm_temperature,
            logprobs=True,
            top_logprobs=min(len(state_node.valid_actions), 20)
        )
        
        text = res.choices[0].message.content

        top_logprobs = res.choices[0].logprobs.content[0].top_logprobs
        
        action_log_dict = {}
        for i, logprob in enumerate(top_logprobs):
            action_token = logprob.token
            action_logprob = logprob.logprob
            action_log_dict[action_token] = action_logprob

        logprobs_list = [action_log_dict.get(label, -5) for label in valid_labels]
        probs_list = softmax(logprobs_list, self.softmax_temperature)
        
        return text, probs_list


    def get_traj_reflection(self, trajectory: List[Dict]) -> str:
        sys_prompt, prompt = self.get_reflection_prompts(trajectory)
        res = chat_completion_with_retries(
            model=self.model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=64,
            temperature=self.llm_temperature
        )
        text = res.choices[0].message.content
        print(text)
        return text
