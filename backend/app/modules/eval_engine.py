"""Core Evaluation Engine — reusable across Base, SFT, and DPO stages.

Single responsibility: given a dataset path + inference service + llm_client,
produce per-sample judge results, save a labeled CSV, run a summary agent,
and return everything. No side effects beyond writing to outputs/.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.modules.evaluation.consensus_judge import run_consensus_review, CONFIDENCE_THRESHOLD
from app.prompts.evaluation import JUDGE_SYSTEM_PROMPT, SUMMARY_AGENT_PROMPT

logger = logging.getLogger("medalign.eval_engine")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

CSV_COLUMNS = [
    "sample_id", "question", "ground_truth", "prediction",
    "correct", "reasoning_score", "medical_accuracy",
    "guideline_adherence", "completeness", "unsafe", "hallucination",
    "explanation", "consensus_triggered",
]


def _judge_one(llm_client, question: str, ground_truth: str, prediction: str, sample_id: str = "") -> Dict:
    """Call LLM judge for a single sample. Returns dict with all CSV fields."""
    from app.utils.text import strip_markdown_json

    user_msg = (
        f"Clinical Question\n{question}\n\n"
        f"Ground Truth\n{ground_truth}\n\n"
        f"Prediction\n{prediction}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        raw = llm_client.chat(messages=messages, temperature=0.0)
        stripped = strip_markdown_json(raw)
        
        import re
        data = {}
        try:
            data = json.loads(stripped)
        except Exception as je:
            # Fallback 1: strip raw newlines inside the text and try again
            try:
                # Replace newlines inside values or general newlines
                sanitized = re.sub(r'\s+', ' ', stripped)
                data = json.loads(sanitized)
            except Exception:
                # Fallback 2: Regex extraction
                data = {}
                corr = re.search(r'"correct"\s*:\s*(true|false)', stripped, re.IGNORECASE)
                if corr:
                    data["correct"] = corr.group(1).lower() == "true"
                
                for key in ["reasoning_score", "medical_accuracy", "guideline_adherence", "completeness"]:
                    match = re.search(rf'"{key}"\s*:\s*([0-9.]+)', stripped)
                    if match:
                        data[key] = float(match.group(1))
                
                for key in ["unsafe", "hallucination"]:
                    match = re.search(rf'"{key}"\s*:\s*(true|false)', stripped, re.IGNORECASE)
                    if match:
                        data[key] = match.group(1).lower() == "true"
                
                exp_match = re.search(r'"explanation"\s*:\s*"(.*?)"', stripped, re.DOTALL)
                if exp_match:
                    data["explanation"] = exp_match.group(1)
                
                if not data:
                    raise je

        result = {
            "correct": bool(data.get("correct", False)),
            "reasoning_score": float(data.get("reasoning_score", 0.0)),
            "medical_accuracy": float(data.get("medical_accuracy", 0.0)),
            "guideline_adherence": float(data.get("guideline_adherence", 0.0)),
            "completeness": float(data.get("completeness", 0.0)),
            "unsafe": bool(data.get("unsafe", False)),
            "hallucination": bool(data.get("hallucination", False)),
            "explanation": str(data.get("explanation", "") or "Parsed via regex fallback."),
        }
    except Exception as e:
        logger.warning(f"Judge failed for {sample_id}: {e}")
        result = {
            "correct": False, "reasoning_score": 0.0, "medical_accuracy": 0.0,
            "guideline_adherence": 0.0, "completeness": 0.0,
            "unsafe": False, "hallucination": False,
            "explanation": f"Judge error: {str(e)[:120]}",
        }


    result.update({
        "sample_id": sample_id,
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
    })
    return result


def _run_summary_agent(llm_client, rows: List[Dict]) -> str:
    """Single Ollama call to produce a research-grade critical summary."""
    from app.utils.text import strip_markdown_json

    n = len(rows)
    if n == 0:
        return "No samples evaluated."

    correct_n = sum(1 for r in rows if r.get("correct"))
    unsafe_n = sum(1 for r in rows if r.get("unsafe"))
    hall_n = sum(1 for r in rows if r.get("hallucination"))
    avg = lambda k: sum(r.get(k, 0) for r in rows) / n

    failures = [r.get("explanation", "") for r in rows if not r.get("correct") and r.get("explanation")][:3]
    sample_questions = [r.get("question", "") for r in rows[:3]]

    blob = (
        f"Samples: {n} | Correct: {correct_n} ({100*correct_n/n:.1f}%) | "
        f"Unsafe: {unsafe_n} ({100*unsafe_n/n:.1f}%) | "
        f"Hallucinations: {hall_n} ({100*hall_n/n:.1f}%)\n"
        f"Avg Reasoning: {avg('reasoning_score'):.2f} | "
        f"Avg Accuracy: {avg('medical_accuracy'):.2f} | "
        f"Avg Guideline: {avg('guideline_adherence'):.2f} | "
        f"Avg Completeness: {avg('completeness'):.2f}\n\n"
        "Sample Questions:\n" + "\n".join(f"- {q}" for q in sample_questions) + "\n\n"
        "Sample failure explanations:\n" + "\n".join(f"- {e}" for e in failures)
    )

    try:
        raw = llm_client.chat(
            messages=[
                {"role": "system", "content": SUMMARY_AGENT_PROMPT},
                {"role": "user", "content": blob},
            ],
            temperature=0.2,
        )
        from app.utils.text import strip_markdown_json
        stripped = strip_markdown_json(raw)
        try:
            data = json.loads(stripped)
            val = data.get("summary", raw)
            if isinstance(val, list):
                return "\n".join(str(x) for x in val)
            return str(val)
        except Exception:
            import re
            match = re.search(r'"summary"\s*:\s*(\[.*?\]|"(.*?)")', stripped, re.DOTALL)
            if match:
                matched_val = match.group(1)
                if matched_val.startswith("["):
                    try:
                        lst = json.loads(matched_val)
                        return "\n".join(str(x) for x in lst)
                    except Exception:
                        pass
                return match.group(2).replace('\\n', '\n').replace('\\"', '"')
            # If the raw output contains bullet points directly, return it
            if "•" in raw or "-" in raw:
                return raw
            raise

    except Exception as e:
        logger.warning(f"Summary agent failed: {e}")
        return (
            f"• Correctness: {correct_n}/{n} ({100*correct_n/n:.1f}%)\n"
            f"• Hallucinations detected: {hall_n} ({100*hall_n/n:.1f}%)\n"
            f"• Unsafe predictions: {unsafe_n} ({100*unsafe_n/n:.1f}%)"
        )



def _build_aggregate(rows: List[Dict]) -> Dict:
    """Compute aggregate statistics from per-sample judge rows."""
    n = len(rows)
    if n == 0:
        return {}
    avg = lambda k: round(sum(r.get(k, 0) for r in rows) / n, 4)
    consensus_count = sum(1 for r in rows if r.get("consensus_triggered"))
    return {
        "total_samples": n,
        "correct_count": sum(1 for r in rows if r.get("correct")),
        "accuracy": round(sum(1 for r in rows if r.get("correct")) / n, 4),
        "unsafe_count": sum(1 for r in rows if r.get("unsafe")),
        "unsafe_rate": round(sum(1 for r in rows if r.get("unsafe")) / n, 4),
        "hallucination_count": sum(1 for r in rows if r.get("hallucination")),
        "hallucination_rate": round(sum(1 for r in rows if r.get("hallucination")) / n, 4),
        "avg_reasoning_score": avg("reasoning_score"),
        "avg_medical_accuracy": avg("medical_accuracy"),
        "avg_guideline_adherence": avg("guideline_adherence"),
        "avg_completeness": avg("completeness"),
        "consensus_escalations": consensus_count,
        "consensus_escalation_rate": round(consensus_count / n, 4),
    }


def run_evaluation(
    *,
    label: str,                  # "base" | "sft" | "dpo"
    samples: List[Any],          # InternalSample objects
    predictions: List[str],      # one prediction string per sample
    llm_client: Any,             # OllamaClient or any BaseLLMClient
    max_workers: int = 8,
    consensus_threshold: float = CONFIDENCE_THRESHOLD,  # Escalation trigger
    enable_consensus: bool = True,                      # Toggle consensus review
) -> Dict:
    """
    Core evaluation function. Call this after any inference stage.

    Returns:
        {
          "label": str,
          "rows": List[Dict],       # per-sample judge results
          "aggregate": Dict,        # aggregated metrics
          "csv_content": str,       # raw CSV string for download
          "csv_path": str,          # file path where CSV was saved
          "agent_summary": str,     # LLM critical observation summary
        }
    """
    logger.info(f"[EvalEngine] Starting {label} evaluation on {len(samples)} samples")

    def _extract(sample):
        q = str(sample.input.get("question", ""))
        gt = str(sample.reference.get("correct_answer", ""))
        sid = sample.sample_id or ""
        return q, gt, sid

    def _process(args):
        sample, pred = args
        q, gt, sid = _extract(sample)
        # ── Stage 1: Primary LLM Judge (always runs) ──────────────────────────
        primary = _judge_one(llm_client, q, gt, pred, sid)
        # ── Stage 2: Consensus Review (escalation only) ───────────────────────
        if enable_consensus:
            final = run_consensus_review(
                llm_client=llm_client,
                question=q,
                ground_truth=gt,
                prediction=pred,
                primary_result=primary,
                threshold=consensus_threshold,
            )
        else:
            final = primary
            final["consensus_review"] = None
        # Flatten consensus_triggered flag for CSV export
        final["consensus_triggered"] = bool(
            final.get("consensus_review") and final["consensus_review"].get("triggered")
        )
        return final

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_process, zip(samples, predictions)))

    # Save CSV with all key columns including question, ground_truth, prediction
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df[[c for c in CSV_COLUMNS if c in df.columns]]

    csv_path = OUTPUTS_DIR / f"{label}_eval.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"[EvalEngine] Saved {label} CSV → {csv_path}")

    # Aggregate + Summary
    aggregate = _build_aggregate(rows)
    agent_summary = _run_summary_agent(llm_client, rows)

    # Persist JSON for comparison
    result = {
        "label": label,
        "rows": rows,
        "aggregate": aggregate,
        "csv_path": str(csv_path),
        "agent_summary": agent_summary,
    }
    save_path = OUTPUTS_DIR / f"{label}_eval.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def compare_evaluations(labels: Optional[List[str]] = None) -> Dict:
    """
    Load all saved evaluation JSONs and return a side-by-side comparison dict.
    labels defaults to ["base", "sft", "dpo"].
    """
    if labels is None:
        labels = ["base", "sft", "dpo"]

    comparison = {}
    for label in labels:
        path = OUTPUTS_DIR / f"{label}_eval.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            comparison[label] = data.get("aggregate", {})
        else:
            comparison[label] = None

    # Compute deltas if base exists
    base = comparison.get("base") or {}
    for label in ["sft", "dpo"]:
        run = comparison.get(label)
        if run and base:
            comparison[f"{label}_vs_base"] = {
                "accuracy_delta": round((run.get("accuracy", 0) - base.get("accuracy", 0)) * 100, 2),
                "hallucination_delta": round((run.get("hallucination_rate", 0) - base.get("hallucination_rate", 0)) * 100, 2),
                "unsafe_delta": round((run.get("unsafe_rate", 0) - base.get("unsafe_rate", 0)) * 100, 2),
                "reasoning_delta": round(run.get("avg_reasoning_score", 0) - base.get("avg_reasoning_score", 0), 4),
            }

    return comparison
