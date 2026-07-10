"""Ollama LLM client implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.services.llm.base import BaseLLMClient
from app.utils.logging import get_logger, log_execution

logger = get_logger("ollama_client")

try:
    import requests
except ImportError:
    requests = None


class OllamaClient(BaseLLMClient):
    """Client for local Ollama models (e.g. llama3.2:latest)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        fallback_models: List[str] = None
    ) -> None:
        if requests is None:
            raise ImportError("The 'requests' library is required to use OllamaClient.")
        self.base_url = base_url.rstrip("/")
        self.fallback_models = fallback_models or ["llama3.2:latest"]

    @property
    def provider(self) -> str:
        return "ollama"

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        response_schema: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Call Ollama chat completion API, trying fallback models in sequence if one fails."""
        
        # If a specific model is requested via kwargs, just try that one.
        # Otherwise, try the fallback models in order.
        models_to_try = [kwargs["model"]] if "model" in kwargs else self.fallback_models
        
        format_param = None
        
        # We need a copy of messages because we might mutate it, and we don't want to double-mutate on retries
        current_messages = list(messages)
        
        if response_schema:
            format_param = "json"
            try:
                schema_str = json.dumps(response_schema.model_json_schema())
                json_instruction = (
                    f"\\nYou must output your response in valid JSON matching exactly "
                    f"this schema: {schema_str}\\nReturn ONLY the raw JSON object."
                )
            except Exception:
                json_instruction = "\\nYou must output ONLY valid JSON."
                
            if current_messages and current_messages[0]["role"] == "system":
                # Copy the dict before mutating
                new_system = dict(current_messages[0])
                new_system["content"] += json_instruction
                current_messages[0] = new_system
            else:
                current_messages.insert(0, {"role": "system", "content": json_instruction})

        last_exception = None

        with log_execution(logger, "ollama_chat_completion"):
            for model_name in models_to_try:
                payload = {
                    "model": model_name,
                    "messages": current_messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 4096
                    }
                }
                
                if format_param:
                    payload["format"] = format_param

                try:
                    logger.info(f"Attempting Ollama inference with model: {model_name}")
                    response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=1200)
                    response.raise_for_status()
                    data = response.json()
                    
                    content = data.get("message", {}).get("content", "")
                    return content
                except Exception as e:
                    logger.warning(f"Ollama API error with model {model_name}: {e}")
                    last_exception = e
                    continue # Try the next model

        # If we exhausted the list, raise the last error
        logger.error(f"All fallback models failed. Last error: {last_exception}")
        raise last_exception if last_exception else Exception("No fallback models available to try.")
