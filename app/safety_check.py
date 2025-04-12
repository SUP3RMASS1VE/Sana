# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/mit-han-lab/hart/blob/main/hart/utils/safety_check.py
import json
from functools import lru_cache
from typing import Dict, Sequence

import torch
import transformers
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

__all__ = ["is_dangerous"]

__rank, __local_rank, __world_size, __device = (
    0,
    0,
    1,
    "cuda" if torch.cuda.is_available() else "cpu",
)


def is_dangerous(tokenizer, model, user_prompt, threshold=0.2):
    """
    Check if a prompt is potentially unsafe using the RM-Gemma-2B reward model.
    
    Args:
        tokenizer: The tokenizer for the model
        model: The RM-Gemma-2B model
        user_prompt: The user prompt to check
        threshold: Threshold below which prompts are considered unsafe
        
    Returns:
        bool: True if the prompt is potentially unsafe, False otherwise
    """
    # For RM-Gemma-2B, we need to format the input as a chat
    chat = [
        {"role": "user", "content": user_prompt.strip()},
    ]
    
    # Create a sentiment analysis pipeline if it's not already a pipeline
    if not isinstance(model, transformers.pipelines.base.Pipeline):
        rm_pipe = pipeline(
            "sentiment-analysis",
            model=model,
            device=model.device,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
    else:
        rm_pipe = model
    
    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1
    }
    
    try:
        # Apply chat template to format the input correctly for the model
        test_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        pipe_output = rm_pipe([test_text], **pipe_kwargs)
        
        # RM-Gemma-2B outputs a score where higher is safer
        safety_score = pipe_output[0][0]["score"]
        
        # Lower score indicates potentially unsafe content
        # The threshold is set to consider prompts with scores below it as unsafe
        return safety_score < threshold
    except Exception as e:
        print(f"Error in safety check: {e}")
        # If there's an error, default to safe behavior
        return False
