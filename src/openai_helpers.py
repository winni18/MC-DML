import time
from typing import Dict, List, Mapping, Optional
import openai
from openai import OpenAI
import tiktoken


client = OpenAI(api_key='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')
encoding = tiktoken.get_encoding("cl100k_base")



def chat_completion_with_retries(model: str, sys_prompt: str, prompt: str, max_retries: int = 5, retry_interval_sec: int = 20, api_keys=None, **kwargs) -> Mapping:

    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            res = client.chat.completions.create(model=model,
            messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            **kwargs)
            
            return res

        except (
            openai.RateLimitError,
            openai.APIError,
            openai.OpenAIError,
            ) as e:
            print(e)
            print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)
    return {}



def truncate_text(text, max_tokens):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        print(f"WARNING: Maximum token length exceeded ({len(tokens)} > {max_tokens})")
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

