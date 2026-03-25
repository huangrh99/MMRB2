# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Local model management for running evaluators on GPU."""

from typing import List, Optional

import torch
from PIL import Image
from transformers import pipeline


class LocalModelManager:
    """Manager for loading and running local VLM models."""

    MODEL_CONFIGS = {
        "qwen3-vl-8b": {
            "huggingface_id": "Qwen/Qwen3-VL-8B-Instruct",
        },
        "qwen3.5-0.8b": {
            "huggingface_id": "Qwen/Qwen3.5-0.8B",
        },
        "qwen3.5-2b": {
            "huggingface_id": "Qwen/Qwen3.5-2B",
        },
        "qwen3.5-4b": {
            "huggingface_id": "Qwen/Qwen3.5-4B",
        },
        "qwen3.5-9b": {
            "huggingface_id": "Qwen/Qwen3.5-9B",
        },
        "qwen3.5-27b": {
            "huggingface_id": "Qwen/Qwen3.5-27B",
            "tp": True,
        },
        "qwen3.5-35b-a3b": {
            "huggingface_id": "Qwen/Qwen3.5-35B-A3B",
            "tp": True,
        },
        "qwen3.5-122b-a10b": {
            "huggingface_id": "Qwen/Qwen3.5-122B-A10B",
            "tp": True,
        },
        "qwen3.5-397b-a17b": {
            "huggingface_id": "Qwen/Qwen3.5-397B-A17B",
            "tp": True,
        },
        # InternVL3.5 (-HF = native HuggingFace transformers pipeline support)
        "internvl3.5-1b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-1B-HF",
        },
        "internvl3.5-2b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-2B-HF",
        },
        "internvl3.5-4b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-4B-HF",
        },
        "internvl3.5-8b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-8B-HF",
        },
        "internvl3.5-14b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-14B-HF",
        },
        "internvl3.5-38b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-38B-HF",
            "tp": True,
        },
        "internvl3.5-20b-a4b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF",
        },
        "internvl3.5-30b-a3b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-30B-A3B-HF",
        },
        "internvl3.5-241b-a28b": {
            "huggingface_id": "OpenGVLab/InternVL3_5-241B-A28B-HF",
            "tp": True,
        },
    }

    def __init__(
        self,
        model_name: str,
        device_id: Optional[int] = None,
    ):
        """Initialize the local model manager.

        Args:
            model_name: Name of the model to load.
            device_id: GPU device ID to use. If None, uses CUDA if available.
        """
        self.model_name = model_name

        if device_id is not None:
            self.device_id = device_id
        else:
            self.device_id = "cuda" if torch.cuda.is_available() else "cpu"

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available models: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_full_name = self.MODEL_CONFIGS[model_name]["huggingface_id"]
        self.tp = self.MODEL_CONFIGS[model_name].get("tp", False)

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        }

        if self.tp:
            print("Big model. Use all GPUs.")
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_full_name,
                device_map="auto",
                model_kwargs=model_kwargs,
            )
        else:
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_full_name,
                device=self.device_id,
                model_kwargs=model_kwargs,
            )

    def generate_response(
        self,
        prompt: List[List[str]],
        max_new_tokens: int = 16384,
        temperature: float = 0.6,
    ) -> str:
        """Generate a response from the model.

        Args:
            prompt: List of [type, content] pairs where type is "text" or "image".
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        system_prompt = None
        messages = []

        if system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        user_content = []
        for segment in prompt:
            if segment[0] == "text":
                user_content.append({"type": "text", "text": segment[1]})
            elif segment[0] == "image":
                user_content.append({"type": "image", "image": Image.open(segment[1])})

        messages.append({"role": "user", "content": user_content})

        response = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=temperature,
        )

        return response[0]["generated_text"]
