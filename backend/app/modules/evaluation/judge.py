"""Medical LLM Judge implementation — compact rubric version."""

from __future__ import annotations

import json
from typing import Any, Dict

from app.modules.evaluation.base import BaseEvaluator
from app.modules.evaluation.prompts import MEDICAL_JUDGE_PROMPT
from app.schemas.judge import MedicalJudgeResponse
from app.services.llm.base import BaseLLMClient
from app.utils.exceptions import EvaluationError
from app.utils.logging import get_logger, log_execution
from app.utils.text import strip_markdown_json

logger = get_logger("evaluation")


class MedicalLLMJudge(BaseEvaluator):
    """Evaluates medical AI predictions using a compact clinical rubric via LLM."""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    def evaluate(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate a prediction and return the 8-field judge result dict."""

        user_prompt = (
            f"Clinical Question\n{question}\n\n"
            f"--------------------------------\n\n"
            f"Ground Truth\n{ground_truth}\n\n"
            f"--------------------------------\n\n"
            f"Prediction\n{prediction}\n\n"
            f"Evaluate the Prediction against the Ground Truth."
        )

        messages = [
            {"role": "system", "content": MEDICAL_JUDGE_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        with log_execution(logger, "judge_evaluation"):
            try:
                response_text = self.llm_client.chat(
                    messages=messages,
                    temperature=0.0,
                    # No response_schema — compact schema embedded in system prompt
                )

                clean_text = strip_markdown_json(response_text)
                data = json.loads(clean_text)

                # Validate and return
                parsed = MedicalJudgeResponse.model_validate(data)
                return parsed.model_dump()

            except Exception as e:
                logger.error(f"LLM Judge evaluation failed: {e}")
                # Return a zero-scored fallback rather than crashing the pipeline
                return MedicalJudgeResponse(
                    explanation=f"Judge failed: {str(e)[:120]}"
                ).model_dump()
