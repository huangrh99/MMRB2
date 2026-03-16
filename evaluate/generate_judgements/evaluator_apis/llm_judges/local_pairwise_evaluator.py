# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Local model pairwise evaluators."""

import json
import re
from typing import List

import json_repair

from ..base import BasePairwiseEvaluator, EvaluatorResult
from .evaluation_prompts import (
    get_image_edit_prompt,
    get_image_gen_prompt,
    get_interleaved_prompt,
    get_reasoning_prompt,
)
from .local_models import LocalModelManager


class LocalPairwiseEvaluator(BasePairwiseEvaluator):
    """Pairwise evaluator using local VLM models."""

    def __init__(self, model_name: str, device_id: int = None):
        self.model_name = model_name
        self.device_id = device_id
        self.model_manager = LocalModelManager(model_name, device_id)

    @property
    def evaluator_name(self):
        return f"{self.model_name}_pairwise_evaluator"

    def parse_llm_json(self, text):
        """Parse JSON from LLM output that may be wrapped in markdown code blocks.

        Args:
            text: The raw text output from the LLM.

        Returns:
            Parsed JSON as a dictionary.
        """
        # Remove thinking tags (e.g., from Qwen3.5 thinking mode)
        text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text.strip(), flags=re.DOTALL)
        # Remove markdown code block formatting
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
        # Select evaluation prompt based on task type
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

        prompt_text = evaluation_prompt + "\n\n"
        prompt_text += "CRITICAL: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON."

        # Build the content list
        content_list = []
        content_list.append(["text", evaluation_prompt])
        content_list.append(["text", "[ORIGINAL PROMPT TO MODEL:]"])
        content_list.extend(prompt_content)
        content_list.append(["text", "[RESPONSE A:]"])
        content_list.extend(response_a)
        content_list.append(["text", "[RESPONSE B:]"])
        content_list.extend(response_b)

        outputs = []
        max_parse_retries = 3
        for _ in range(n):
            parsed_response = None
            for parse_attempt in range(max_parse_retries):
                response = self.model_manager.generate_response(content_list)
                try:
                    parsed_response = self.parse_llm_json(response)
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


class Qwen3VL8BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3-VL-8B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3-vl-8b", device_id=device_id)


class Qwen35VL08BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3.5-0.8B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3.5-0.8b", device_id=device_id)


class Qwen35VL2BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3.5-2B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3.5-2b", device_id=device_id)


class Qwen35VL4BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3.5-4B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3.5-4b", device_id=device_id)


class Qwen35VL9BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3.5-9B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3.5-9b", device_id=device_id)


class Qwen35VL27BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3.5-27B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3.5-27b", device_id=device_id)


class Qwen35VL35BA3BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3.5-35B-A3B (MoE) based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3.5-35b-a3b", device_id=device_id)


class Qwen35VL122BA10BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3.5-122B-A10B (MoE) based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3.5-122b-a10b", device_id=device_id)


class Qwen35VL397BA17BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3.5-397B-A17B (MoE) based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3.5-397b-a17b", device_id=device_id)


class InternVL35_1BPairwiseEvaluator(LocalPairwiseEvaluator):
    """InternVL3.5-1B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="internvl3.5-1b", device_id=device_id)


class InternVL35_2BPairwiseEvaluator(LocalPairwiseEvaluator):
    """InternVL3.5-2B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="internvl3.5-2b", device_id=device_id)


class InternVL35_4BPairwiseEvaluator(LocalPairwiseEvaluator):
    """InternVL3.5-4B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="internvl3.5-4b", device_id=device_id)


class InternVL35_8BPairwiseEvaluator(LocalPairwiseEvaluator):
    """InternVL3.5-8B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="internvl3.5-8b", device_id=device_id)


class InternVL35_14BPairwiseEvaluator(LocalPairwiseEvaluator):
    """InternVL3.5-14B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="internvl3.5-14b", device_id=device_id)


class InternVL35_38BPairwiseEvaluator(LocalPairwiseEvaluator):
    """InternVL3.5-38B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="internvl3.5-38b", device_id=device_id)
