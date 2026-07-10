import os
import json
import torch
import logging
import zipfile
import shutil
import gc
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

logger = logging.getLogger("train_api")
router = APIRouter()

def zip_directory(dir_path: str, zip_path: str):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, dir_path))

@router.post("/sft")
async def train_sft(file: UploadFile = File(...), params: str = Form(...)):
    try:
        parsed_params = json.loads(params)
        model_name = parsed_params.get("model", "Qwen/Qwen2.5-0.5B")
        lr = float(parsed_params.get("lr", "2e-4"))
        epochs = int(parsed_params.get("epochs", 3))
        batch_size = int(parsed_params.get("batch_size", 4))
        r = int(parsed_params.get("r", 16))

        content = await file.read()
        content_str = content.decode("utf-8").strip()
        try:
            data = json.loads(content_str)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            lines = content_str.split("\n")
            data = [json.loads(line) for line in lines if line.strip()]
        from app.utils.schema import agentic_remap_dataset
        data = agentic_remap_dataset(data, expected_keys=["instruction", "output"])
        train_dataset = Dataset.from_list(data)

        def formatting_func(example):
            inst = example.get("instruction", "")
            out = example.get("output", "")
            return {"text": str(inst) + "\\n" + str(out)}
        
        if "text" not in train_dataset.column_names:
            train_dataset = train_dataset.map(formatting_func)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(r=r, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)

        run_name = parsed_params.get("run_name", "").strip()
        dir_name = f"sft_output_{run_name}" if run_name else "sft_output"
        output_dir = f"/tmp/{dir_name}"
        os.makedirs(output_dir, exist_ok=True)

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            bf16=torch.cuda.is_available(),
            max_length=1024,
            report_to="none"
        )
        
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset, processing_class=tokenizer)
        trainer.train()
        trainer.save_model(output_dir)

        zip_path = f"/tmp/{dir_name}.zip"
        zip_directory(output_dir, zip_path)
        
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        return FileResponse(path=zip_path, filename=f"{dir_name}.zip", media_type='application/zip')
    except Exception as e:
        logger.error(f"SFT training failed: {e}")
        raise HTTPException(status_code=500, detail=f"SFT training failed: {str(e)}")

@router.post("/dpo")
async def train_dpo(file: UploadFile = File(...), params: str = Form(...)):
    try:
        parsed_params = json.loads(params)
        model_name = parsed_params.get("model", "Qwen/Qwen2.5-0.5B")
        lr = float(parsed_params.get("lr", "5e-5"))
        epochs = int(parsed_params.get("epochs", 2))
        batch_size = int(parsed_params.get("batch_size", 2))
        beta = float(parsed_params.get("beta", 0.1))

        content = await file.read()
        content_str = content.decode("utf-8").strip()
        try:
            data = json.loads(content_str)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            try:
                # Attempt to parse line-by-line for strict JSONL
                lines = content_str.split("\n")
                data = [json.loads(line) for line in lines if line.strip()]
            except json.JSONDecodeError:
                # The user likely uploaded pretty-printed JSONL (multiple root objects).
                # Convert it into a valid JSON array by wrapping in [] and separating }{ with },{
                import re
                fixed_str = re.sub(r'}\s*{', '},{', content_str)
                fixed_str = f"[{fixed_str}]"
                data = json.loads(fixed_str)
        
        from app.utils.schema import agentic_remap_dataset
        data = agentic_remap_dataset(data, expected_keys=["prompt", "chosen", "rejected"])
        train_dataset = Dataset.from_list(data)

        actual_model = model_name if model_name != "local-sft-model" else "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(actual_model, trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        
        if model_name == "local-sft-model" and os.path.exists("/tmp/sft_output"):
            # Load base model, then load SFT adapter and make it trainable
            base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", quantization_config=bnb_config, device_map="auto")
            base_model = prepare_model_for_kbit_training(base_model)
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, "/tmp/sft_output", is_trainable=True)
        else:
            # DPO directly on base model
            model = AutoModelForCausalLM.from_pretrained(actual_model, quantization_config=bnb_config, device_map="auto")
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
            model = get_peft_model(model, peft_config)

        run_name = parsed_params.get("run_name", "").strip()
        dir_name = f"dpo_output_{run_name}" if run_name else "dpo_output"
        output_dir = f"/tmp/{dir_name}"
        os.makedirs(output_dir, exist_ok=True)

        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            beta=beta,
            bf16=torch.cuda.is_available(),
            max_length=1024,
            report_to="none",
            remove_unused_columns=False
        )
        
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=train_dataset, processing_class=tokenizer)
        trainer.train()
        trainer.save_model(output_dir)

        zip_path = f"/tmp/{dir_name}.zip"
        zip_directory(output_dir, zip_path)
        
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        return FileResponse(path=zip_path, filename=f"{dir_name}.zip", media_type='application/zip')
    except Exception as e:
        logger.error(f"DPO training failed: {e}")
        raise HTTPException(status_code=500, detail=f"DPO training failed: {str(e)}")
