"""Datasets API endpoints (stub — ready for Prompt 2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.utils.response import api_response
from app.api.v1.deps import get_llm_client
from app.services.llm.base import BaseLLMClient
from app.modules.datasets.synthesizer import DatasetSynthesizer

router = APIRouter()

class AugmentRequest(BaseModel):
    dataset_path: str

@router.get("/")
async def list_datasets() -> dict:
    """List available datasets."""
    return api_response(
        message="Datasets endpoint ready.",
        data={"available": ["medqa", "medmcqa", "pubmedqa"]},
    )

@router.post("/augment")
async def augment_dataset(
    req: AugmentRequest, 
    llm_client: BaseLLMClient = Depends(get_llm_client)
) -> dict:
    """Use the LLM to generate synthetic evaluation data."""
    synthesizer = DatasetSynthesizer(llm_client)
    try:
        result = synthesizer.generate_sft_dataset(req.dataset_path)
        return api_response(
            message=f"Successfully generated {result['generated_count']} SFT pairs.",
            data=result
        )
    except Exception as e:
        return api_response(message=f"Error: {e}", data={})


from fastapi import UploadFile, File
import json
import logging

logger = logging.getLogger("datasets_api")

@router.post("/dpo-convert")
async def convert_dpo_dataset(
    file: UploadFile = File(...),
    llm_client: BaseLLMClient = Depends(get_llm_client)
) -> dict:
    """Convert uploaded SFT dataset JSONL into DPO preference pairs."""
    try:
        content = await file.read()
        content_str = content.decode("utf-8").strip()
        try:
            sft_pairs = json.loads(content_str)
            if not isinstance(sft_pairs, list):
                sft_pairs = [sft_pairs]
        except json.JSONDecodeError:
            lines = content_str.split("\n")
            sft_pairs = []
            for line in lines:
                if line.strip():
                    try:
                        sft_pairs.append(json.loads(line))
                    except Exception:
                        pass
                        
        logger.info(f"Parsed {len(sft_pairs)} pairs from uploaded file. Type: {type(sft_pairs)}")
        
        if not sft_pairs:
            logger.warning("No valid pairs found in uploaded file. Content snippet: " + content_str[:100])
        
        dpo_pairs = []
        
        # Use LLM to generate a rejected response for each SFT instruction/output
        for i, pair in enumerate(sft_pairs):
            # Also support prompt/chosen or question/answer keys
            inst = pair.get("instruction", pair.get("prompt", pair.get("question")))
            out = pair.get("output", pair.get("chosen", pair.get("answer")))
            
            if not inst or not out:
                continue
                
            prompt = (
                f"You are a medical AI researcher. We have a clinical question and a CORRECT medical answer.\n"
                f"Clinical Question:\n{inst}\n\n"
                f"Correct Answer:\n{out}\n\n"
                f"Please generate a plausible but INCORRECT (rejected) answer that a flawed medical model might generate. "
                f"It should sound professional but contain a critical medical error or hallucination.\n\n"
                f"You MUST return your response as a valid JSON object with a single key 'rejected_answer'.\n"
                f"Example: {{\"rejected_answer\": \"The patient should be given ibuprofen...\"}}\n"
                f"Return ONLY the raw JSON object, no markdown, no quotes."
            )
            
            try:
                raw_response = llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8
                ).strip()
                
                # Cleanup common LLM markdown formatting
                if raw_response.startswith('```json'):
                    raw_response = raw_response[7:]
                if raw_response.startswith('```'):
                    raw_response = raw_response[3:]
                if raw_response.endswith('```'):
                    raw_response = raw_response[:-3]
                raw_response = raw_response.strip()
                
                try:
                    parsed = json.loads(raw_response)
                    rejected = parsed.get("rejected_answer", "")
                except json.JSONDecodeError:
                    rejected = raw_response
                    
                if rejected.startswith('"') and rejected.endswith('"'):
                    rejected = rejected[1:-1]
                    
                dpo_pairs.append({
                    "prompt": inst,
                    "chosen": out,
                    "rejected": rejected.strip()
                })
            except Exception as e:
                logger.error(f"Failed to generate DPO pair for sample {i}: {e}")
                
        if not dpo_pairs:
            logger.error(f"No DPO pairs were generated! SFT pairs parsed: {len(sft_pairs)}")
            return api_response(message="Error: No valid instruction/output pairs found in the uploaded file.", data={})
            
        from pathlib import Path
        pref_path = Path("outputs/preference_dataset.jsonl")
        pref_path.parent.mkdir(exist_ok=True, parents=True)
        with open(pref_path, "w") as f:
            for d in dpo_pairs:
                f.write(json.dumps(d) + "\n")
                
        return api_response(
            message=f"Successfully converted {len(dpo_pairs)} SFT pairs to DPO format.",
            data={"preference_dataset": dpo_pairs}
        )
    except Exception as e:
        logger.error(f"Error in DPO conversion: {e}")
        return api_response(message=f"Error: {e}", data={})

