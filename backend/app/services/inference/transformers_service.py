"""Local HuggingFace Transformers inference service."""

from __future__ import annotations

import time
from typing import Any, List

from app.services.inference.base import BaseInferenceService
from app.utils.logging import get_logger
from app.utils.exceptions import InferenceError

logger = get_logger("transformers_inference")


class TransformersInferenceService(BaseInferenceService):
    """Local inference using HuggingFace Transformers."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda" # Default to cuda, transformers will handle fallback if needed or we specify

    def load_model(self, lora_path: str = None, model_name: str = None, **kwargs: Any) -> None:
        """Load the model and tokenizer from HuggingFace."""
        if model_name:
            self.model_id = model_name
        try:
            import torch # type: ignore
            from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
            
            logger.info(f"Loading tokenizer for {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            if self.model is not None:
                import gc
                del self.model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info(f"Loading model {self.model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            if lora_path:
                import os
                if os.path.exists(lora_path):
                    from peft import PeftModel # type: ignore
                    logger.info(f"Loading PEFT adapter from {lora_path}...")
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                else:
                    logger.warning(f"LoRA path {lora_path} does not exist, using base model.")

            # Identify the device the model actually loaded on (for the tokenizer/inputs)
            # device_map="auto" usually puts it on cuda:0 if available
            self.device = getattr(self.model, "device", "cuda")
            logger.info(f"Model loaded successfully on {self.device}.")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise InferenceError(f"Failed to load model: {e}")

    def generate(
        self,
        messages: List[dict[str, str]],
        **kwargs: Any,
    ) -> tuple[str, float]:
        """Generate a completion from chat messages."""
        if self.model is None or self.tokenizer is None:
            logger.info("Model not loaded yet. Loading now...")
            self.load_model()
            
        start_time = time.time()
        
        try:
            # Format messages using the chat template if available, else manual fallback
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = ""
                for msg in messages:
                    prompt += f"{msg['role']}: {msg['content']}\n"
                prompt += "assistant: "

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) # type: ignore
            
            max_new_tokens = kwargs.get("max_new_tokens", 256)
            temperature = kwargs.get("temperature", 0.1)
            
            outputs = self.model.generate( # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.eos_token_id # type: ignore
            )
            
            # Decode only the newly generated tokens
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True) # type: ignore
            
            latency = time.time() - start_time
            return text.strip(), latency
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise InferenceError(f"Generation failed: {e}")
