from app.modules.eval_engine import _extract, _judge_one, run_consensus_review
from app.schemas.common import InternalSample
import pandas as pd

sample = InternalSample(
    task_type="clinical_qa",
    input={"question": "q1"},
    reference={"correct_answer": "a1"},
    sample_id="1"
)

class MockClient:
    def chat(self, messages, temperature=0):
        return '{"correct": true, "reasoning_score": 0.5, "medical_accuracy": 0.5, "guideline_adherence": 0.5, "completeness": 0.5, "unsafe": false, "hallucination": false, "explanation": "test"}'

client = MockClient()
q, gt, sid = str(sample.input.get("question", "")), str(sample.reference.get("correct_answer", "")), str(sample.sample_id)
primary = _judge_one(client, q, gt, "p1", sid)
final = run_consensus_review(client, q, gt, "p1", primary, threshold=0.0)

print("final keys:", final.keys())
