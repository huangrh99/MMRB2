# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pairwise evaluator using SGLang/vLLM OpenAI-compatible API server."""

import base64
import json
import math
import os
import re
from pathlib import Path
from time import sleep
from typing import List

import json_repair
from openai import OpenAI

from ..base import BasePairwiseEvaluator, EvaluatorResult
from .evaluation_prompts import (
    get_image_edit_prompt,
    get_image_gen_prompt,
    get_interleaved_prompt,
    get_reasoning_prompt,
)


class SglangPairwiseEvaluator(BasePairwiseEvaluator):
    """Pairwise evaluator using an OpenAI-compatible API server (SGLang/vLLM).

    The user is responsible for starting the server before running evaluation.
    Example:
        python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --port 8000

    Configuration via environment variables:
        SGLANG_BASE_URL:       API base URL (default: http://localhost:8000/v1)
        SGLANG_MODEL_NAME:     Model name to pass to the API (default: auto-detect)
        SGLANG_API_KEY:        API key if required (default: EMPTY)
        SGLANG_ENABLE_THINKING: Enable thinking mode (default: "true")
    """

    def __init__(self, device_id: int = None, enable_thinking: bool = True):
        # Resolve thinking mode: constructor arg can be overridden by env var
        env_thinking = os.environ.get("SGLANG_ENABLE_THINKING", "").lower()
        if env_thinking in ("false", "0", "no"):
            self.enable_thinking = False
        elif env_thinking in ("true", "1", "yes"):
            self.enable_thinking = True
        else:
            self.enable_thinking = enable_thinking

        self.base_url = os.environ.get(
            "SGLANG_BASE_URL", "http://localhost:8000/v1"
        )
        self._model_name = os.environ.get("SGLANG_MODEL_NAME", "")
        api_key = os.environ.get("SGLANG_API_KEY", "EMPTY")

        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

        # Auto-detect model name from server if not specified
        if not self._model_name:
            try:
                models = self.client.models.list()
                self._model_name = models.data[0].id
                print(f"Auto-detected model: {self._model_name}")
            except Exception:
                self._model_name = "default"
                print(
                    f"Could not auto-detect model, using '{self._model_name}'. "
                    "Set SGLANG_MODEL_NAME to override."
                )

        thinking_str = "on" if self.enable_thinking else "off"
        print(
            f"SGLang evaluator: base_url={self.base_url}, "
            f"model={self._model_name}, thinking={thinking_str}"
        )

    @property
    def evaluator_name(self):
        suffix = "_nothink" if not self.enable_thinking else ""
        return f"sglang_{self._model_name}{suffix}_pairwise_evaluator"

    @staticmethod
    def _encode_image(image_path: str, max_aspect_ratio: float = 199.0) -> dict:
        """Encode a local image to base64 OpenAI content format.

        Images with extreme aspect ratios are padded to fit within
        max_aspect_ratio to avoid server rejection.
        """
        from PIL import Image as PILImage
        import io

        img = PILImage.open(image_path)
        w, h = img.size
        aspect = max(w, h) / max(min(w, h), 1)

        # Pad extreme aspect ratio images to a safe ratio
        if aspect > max_aspect_ratio:
            if w >= h:
                # Ensure new_h makes the ratio <= max_aspect_ratio
                new_h = max(int(math.ceil(w / max_aspect_ratio)), h)
                padded = PILImage.new("RGB", (w, new_h), (0, 0, 0))
                padded.paste(img.convert("RGB"), (0, (new_h - h) // 2))
            else:
                new_w = max(int(math.ceil(h / max_aspect_ratio)), w)
                padded = PILImage.new("RGB", (new_w, h), (0, 0, 0))
                padded.paste(img.convert("RGB"), ((new_w - w) // 2, 0))
            img = padded

        ext = Path(image_path).suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        fmt = "JPEG" if mime == "image/jpeg" else "PNG"

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    def _build_messages(self, content_list: List[List[str]]) -> list:
        """Convert internal content_list to OpenAI chat messages format."""
        parts = []
        for seg_type, seg_value in content_list:
            if seg_type == "text":
                parts.append({"type": "text", "text": seg_value})
            elif seg_type == "image":
                parts.append(self._encode_image(seg_value))
        return [{"role": "user", "content": parts}]

    def _chat_with_retry(self, messages: list, max_retries: int = 3) -> str:
        """Call the API with retry logic."""
        extra_kwargs = {}
        if self.enable_thinking:
            # Qwen3.5 recommended: thinking mode for reasoning tasks
            sampling = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5}
        else:
            # Qwen3.5 recommended: non-thinking mode for general tasks
            sampling = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
            extra_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=16384 if self.enable_thinking else 8192,
                    **sampling,
                    **extra_kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"SGLang API attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                sleep(2 ** attempt)

    def parse_llm_json(self, text):
        """Parse JSON from LLM output, stripping thinking tags and code blocks."""
        text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text.strip(), flags=re.DOTALL)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text.strip())

        try:
            return json_repair.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON: {e}\nText was:\n\n{text}\n\n...ends here...\n\n"
            )

    def pairwise_evaluate(
        self,
        prompt_content: List[List[str]],
        response_a: List[List[str]],
        response_b: List[List[str]],
        task_type,
        n: int = 1,
        verbose: bool = False,
    ) -> List[EvaluatorResult]:
        """Evaluate two responses and return judgements."""
        if task_type == "image":
            evaluation_prompt = get_image_gen_prompt()
        elif task_type == "edit":
            evaluation_prompt = get_image_edit_prompt()
        elif task_type == "interleaved":
            evaluation_prompt = get_interleaved_prompt()
        elif task_type in ("text", "reasoning"):
            evaluation_prompt = get_reasoning_prompt()
        else:
            raise ValueError(
                f"Invalid task type: {task_type}. Must be image, edit, interleaved, or reasoning."
            )

        content_list = []
        content_list.append(["text", evaluation_prompt])
        content_list.append(["text", "[ORIGINAL PROMPT TO MODEL:]"])
        content_list.extend(prompt_content)
        content_list.append(["text", "[RESPONSE A:]"])
        content_list.extend(response_a)
        content_list.append(["text", "[RESPONSE B:]"])
        content_list.extend(response_b)

        messages = self._build_messages(content_list)

        outputs = []
        max_parse_retries = 3
        for _ in range(n):
            parsed_response = None
            for parse_attempt in range(max_parse_retries):
                response_text = self._chat_with_retry(messages)
                try:
                    parsed_response = self.parse_llm_json(response_text)
                except ValueError:
                    print(f"Parse attempt {parse_attempt + 1}/{max_parse_retries} failed")
                    continue

                if isinstance(parsed_response, list):
                    parsed_response = next(
                        (x for x in parsed_response if isinstance(x, dict)), None
                    )
                if isinstance(parsed_response, dict) and "better_response" in parsed_response:
                    break
                else:
                    print(
                        f"Parse attempt {parse_attempt + 1}/{max_parse_retries}: "
                        f"missing 'better_response' in {type(parsed_response).__name__}"
                    )
                    parsed_response = None

            if not isinstance(parsed_response, dict) or "better_response" not in parsed_response:
                raise ValueError("All parse retries exhausted. Model could not produce valid JSON.")

            outputs.append(
                EvaluatorResult(
                    judgement=parsed_response["better_response"],
                    metadata=parsed_response,
                )
            )

        return outputs


class SglangNoThinkPairwiseEvaluator(SglangPairwiseEvaluator):
    """SGLang evaluator with thinking mode disabled.

    Passes ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``
    to the API so models like Qwen3.5 run in non-thinking mode.
    """

    def __init__(self, device_id: int = None):
        super().__init__(device_id=device_id, enable_thinking=False)
