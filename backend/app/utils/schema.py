from typing import List, Dict, Any
import json
import logging
from app.services.llm.ollama_client import OllamaClient
from app.config.settings import get_settings
from app.utils.text import strip_markdown_json

logger = logging.getLogger("schema_mapper")

def agentic_remap_dataset(data: List[Dict[str, Any]], expected_keys: List[str] = ["instruction", "output"]) -> List[Dict[str, Any]]:
    """
    Uses an LLM to understand an arbitrary dataset's schema and maps it to the expected keys.
    Only evaluates the first row to determine the mapping, then applies it universally.
    """
    if not data:
        return data

    first_row = data[0]
    
    # Fast path: If data already matches expected format
    if all(k in first_row for k in expected_keys):
        return data

    settings = get_settings()
    llm_client = OllamaClient(base_url=settings.inference.ollama_base_url, fallback_models=settings.inference.fallback_models_list)

    prompt = f"""
    You are an expert data engineer. 
    I have a dataset row that needs to be mapped to the standard keys: {expected_keys}.
    
    Here is the sample row:
    {json.dumps(first_row, indent=2)}
    
    Please map the keys from this row to the expected keys.
    Respond ONLY with a valid JSON object mapping the EXPECTED keys to the ACTUAL keys found in the row.
    For example: {{"instruction": "actual_question_key", "output": "actual_answer_key"}}
    """

    try:
        raw_response = llm_client.chat([
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        
        mapping = json.loads(strip_markdown_json(raw_response))
        logger.info(f"Agentic Schema Mapper deduced mapping: {mapping}")
        
        # Apply mapping
        remapped_data = []
        for row in data:
            new_row = dict(row) # Copy original
            for exp_key, act_key in mapping.items():
                if act_key in row:
                    new_row[exp_key] = row[act_key]
            remapped_data.append(new_row)
            
        return remapped_data

    except Exception as e:
        logger.error(f"Agentic schema mapping failed: {e}. Falling back to original data.")
        return data
