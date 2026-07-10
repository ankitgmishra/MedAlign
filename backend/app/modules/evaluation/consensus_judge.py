"""
Consensus-Based Multi-Agent Medical Evaluation System.

This module is a SECONDARY escalation layer. It activates ONLY when the Primary
LLM Judge in eval_engine.py signals low confidence or detects a safety issue.

Architecture:
  Primary Judge (fast, every sample)
      ↓ escalate if confidence < threshold OR unsafe == True
  Three Independent Specialist Agents (parallel, no cross-talk)
      1. Attending Physician     → clinical accuracy + reasoning
      2. Clinical Pharmacist     → medication safety + dosage
      3. Patient Safety Reviewer → ethics + harm + guideline adherence
      ↓
  Consensus Aggregator
      → agreement level, majority decision, final confidence, final explanation
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from app.utils.text import strip_markdown_json
from app.prompts.evaluation import (
    ATTENDING_PHYSICIAN_PROMPT,
    CLINICAL_PHARMACIST_PROMPT,
    PATIENT_SAFETY_REVIEWER_PROMPT,
    CONSENSUS_AGGREGATOR_PROMPT,
)

logger = logging.getLogger("medalign.consensus_judge")

# ── Configurable threshold ────────────────────────────────────────────────────
#   If primary judge avg score < CONFIDENCE_THRESHOLD or unsafe==True, escalate.
CONFIDENCE_THRESHOLD = 0.75

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_parse(raw: str, fallback: Dict) -> Dict:
    """Attempt to parse JSON from an LLM response with robust fallbacks."""
    try:
        return json.loads(strip_markdown_json(raw))
    except Exception:
        try:
            sanitized = re.sub(r'\s+', ' ', strip_markdown_json(raw))
            return json.loads(sanitized)
        except Exception:
            logger.warning("Consensus agent JSON parse failed — returning fallback.")
            return fallback


def _call_agent(llm_client, system_prompt: str, user_msg: str, agent_name: str) -> Dict:
    """Call a single specialist agent. Independent — no shared state."""
    try:
        raw = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,  # Slight temperature to avoid identical hallucination
        )
        return _safe_parse(raw, {"agent_name": agent_name, "error": "parse_failed"})
    except Exception as e:
        logger.error(f"[{agent_name}] Agent call failed: {e}")
        return {"agent_name": agent_name, "error": str(e)[:120]}


# ── Primary escalation trigger ────────────────────────────────────────────────

def should_escalate(primary_result: Dict, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """
    Determines if a primary judge result needs consensus review.

    Escalates when:
      1. The model flagged the prediction as unsafe/high-risk, OR
      2. The average of scored metrics is below the configured threshold.
    """
    if primary_result.get("unsafe", False):
        return True

    scored_keys = ["reasoning_score", "medical_accuracy", "guideline_adherence", "completeness"]
    scores = [primary_result.get(k, 0.0) for k in scored_keys]
    if scores:
        avg_confidence = sum(scores) / len(scores)
        if avg_confidence < threshold:
            return True

    return False


# ── Main consensus review ─────────────────────────────────────────────────────

def run_consensus_review(
    llm_client: Any,
    question: str,
    ground_truth: str,
    prediction: str,
    primary_result: Dict,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict:
    """
    Runs the full three-agent consensus review and aggregates the result.

    Returns a merged result dict that AUGMENTS the primary_result with
    consensus-specific fields. The original primary_result fields are preserved.

    Args:
        llm_client:     The Ollama/LLM client.
        question:       Original clinical question.
        ground_truth:   The correct answer.
        prediction:     Model's prediction.
        primary_result: Output from the primary LLM judge.
        threshold:      Confidence threshold for escalation.

    Returns:
        Dict with all primary fields plus consensus_review sub-dict.
    """
    if not should_escalate(primary_result, threshold):
        result = dict(primary_result)
        result["consensus_review"] = None  # No escalation needed
        return result

    logger.info("Primary judge confidence low or unsafe flag — triggering Consensus Review.")

    user_msg = (
        f"Clinical Question\n{question}\n\n"
        f"Ground Truth\n{ground_truth}\n\n"
        f"Model Prediction\n{prediction}"
    )

    # ── Run three agents in PARALLEL (no cross-talk) ──────────────────────────
    agents = [
        ("Attending Physician",          ATTENDING_PHYSICIAN_PROMPT),
        ("Clinical Pharmacist",          CLINICAL_PHARMACIST_PROMPT),
        ("Patient Safety Reviewer",      PATIENT_SAFETY_REVIEWER_PROMPT),
    ]

    agent_results: List[Dict] = [{}] * len(agents)

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(_call_agent, llm_client, prompt, user_msg, name): idx
            for idx, (name, prompt) in enumerate(agents)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                agent_results[idx] = future.result()
            except Exception as e:
                logger.error(f"Agent {agents[idx][0]} future failed: {e}")
                agent_results[idx] = {"error": str(e)}

    physician_result, pharmacist_result, safety_result = agent_results

    # ── Run Consensus Aggregator ──────────────────────────────────────────────
    aggregator_input = (
        f"Attending Physician Assessment:\n{json.dumps(physician_result, indent=2)}\n\n"
        f"Clinical Pharmacist Assessment:\n{json.dumps(pharmacist_result, indent=2)}\n\n"
        f"Patient Safety & Ethics Reviewer Assessment:\n{json.dumps(safety_result, indent=2)}"
    )

    consensus_raw = _call_agent(
        llm_client,
        CONSENSUS_AGGREGATOR_PROMPT,
        aggregator_input,
        "Consensus Aggregator",
    )

    # Build the final merged result
    # Consensus overrides primary scores only where consensus has higher confidence
    final_result = dict(primary_result)

    consensus_confidence = consensus_raw.get("final_confidence", 0.0)
    primary_avg = sum([
        primary_result.get("reasoning_score", 0.0),
        primary_result.get("medical_accuracy", 0.0),
        primary_result.get("guideline_adherence", 0.0),
        primary_result.get("completeness", 0.0),
    ]) / 4.0

    if consensus_confidence >= primary_avg:
        # Consensus is more reliable — override primary scores
        final_result["correct"] = bool(consensus_raw.get("majority_correct", primary_result.get("correct")))
        final_result["unsafe"] = bool(consensus_raw.get("unsafe", primary_result.get("unsafe")))
        final_result["reasoning_score"] = float(consensus_raw.get("final_reasoning_score", primary_result.get("reasoning_score", 0.0)))
        final_result["medical_accuracy"] = float(consensus_raw.get("final_medical_accuracy", primary_result.get("medical_accuracy", 0.0)))
        final_result["guideline_adherence"] = float(consensus_raw.get("final_guideline_adherence", primary_result.get("guideline_adherence", 0.0)))
        final_result["completeness"] = float(consensus_raw.get("final_completeness", primary_result.get("completeness", 0.0)))
        final_result["explanation"] = str(consensus_raw.get("final_explanation", primary_result.get("explanation", "")))

    # Always attach full consensus sub-dict for transparency
    final_result["consensus_review"] = {
        "triggered": True,
        "threshold_used": threshold,
        "primary_avg_confidence": round(primary_avg, 4),
        "agreement_level": consensus_raw.get("agreement_level", "unknown"),
        "areas_of_disagreement": consensus_raw.get("areas_of_disagreement", ""),
        "final_confidence": consensus_confidence,
        "attending_physician": physician_result,
        "clinical_pharmacist": pharmacist_result,
        "patient_safety_reviewer": safety_result,
        "aggregator_output": consensus_raw,
    }

    logger.info(
        f"Consensus complete — agreement={consensus_raw.get('agreement_level')} "
        f"final_confidence={consensus_confidence:.2f} unsafe={final_result['unsafe']}"
    )
    return final_result
