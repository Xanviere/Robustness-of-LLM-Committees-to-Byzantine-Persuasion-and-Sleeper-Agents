import asyncio
from typing import Dict, Any, Optional
from core.client_wrapper import LLMClient
from attacks.strategies import EVIDENCE_VERIFIER_SYSTEM_PROMPT

class LLMJudgeVerifier:
    """Exploratory defense: LLM-as-a-judge for evaluating revision proofs."""
    def __init__(self, client: LLMClient):
        self.client = client

    async def evaluate_revision_evidence(
        self,
        task_prompt: str,
        anchor_answer: str,
        proposed_answer: str,
        revision_evidence: str
    ) -> Dict[str, Any]:
        prompt = (
            f"TASK PROMPT:\n{task_prompt}\n\n"
            f"ROUND 0 ANCHOR ANSWER:\n{anchor_answer}\n\n"
            f"PROPOSED REVISED ANSWER:\n{proposed_answer}\n\n"
            f"SUPPLIED REVISION EVIDENCE & DERIVATION:\n{revision_evidence}\n\n"
            "Evaluate whether this evidence logically and mathematically invalidates the anchor and proves the proposed answer."
        )

        res = await self.client.generate(
            prompt=prompt,
            system_instruction=EVIDENCE_VERIFIER_SYSTEM_PROMPT,
            temperature=0.0,
            is_verifier_judge=True
        )

        payload = res.get("payload", {})
        return {
            "evidence_valid": payload.get("evidence_valid", False),
            "score": payload.get("score", 0.0),
            "recommended_gate": payload.get("recommended_gate", "reject_revision"),
            "checkable_claims": payload.get("checkable_claims", []),
            "logical_contradictions": payload.get("logical_contradictions", [])
        }
