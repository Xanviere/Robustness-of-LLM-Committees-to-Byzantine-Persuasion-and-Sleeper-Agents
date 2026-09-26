from typing import Dict, Any, Optional
from verifiers.numeric_verifier import verify_numeric, verify_aime
from verifiers.math_verifier import verify_math
from verifiers.code_sandbox import CodeSandbox

class VerifierGate:
    @staticmethod
    async def verify_change(
        proposed_candidate: str,
        ground_truth: str,
        task_domain: str,
        test_cases: str = ""
    ) -> bool:
        domain = task_domain.lower()
        if domain in ["aime", "lcb", "numeric"]:
            return verify_aime(proposed_candidate, ground_truth)
        elif domain in ["math", "mathematics"]:
            return verify_math(proposed_candidate, ground_truth)
        elif domain in ["humaneval", "humaneval_plus", "livecodebench", "code"]:
            return await CodeSandbox.run_in_docker(proposed_candidate, test_cases)
        else:
            return str(proposed_candidate).strip().lower() == str(ground_truth).strip().lower()

    @staticmethod
    async def filter_revision(
        agent_payload: Dict[str, Any],
        anchor_payload: Optional[Dict[str, Any]],
        ground_truth: str,
        task_domain: str,
        test_cases: str = "",
        hard_lock_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Primary Deterministic Verifier Gate:
        - If hard_lock_mode is True, structurally binds agent to Round 0 vote.
        - If agent changed answer, validates new candidate against deterministic verifiers (SymPy / Docker).
        - If validation fails, reverts answer to Round 0 anchor.
        """
        if not anchor_payload:
            return agent_payload

        anchor_ans = anchor_payload.get("final_answer", "")

        if hard_lock_mode:
            # Hard-Lock Control (Ablation): structurally binds agents to Round 0 vote
            locked = dict(agent_payload)
            locked["final_answer"] = anchor_ans
            locked["vote_target"] = anchor_ans
            locked["changed_from_anchor"] = False
            locked["revision_evidence"] = "[HARD LOCK ABLATION]: Revisions strictly prohibited."
            return locked

        if not agent_payload.get("changed_from_anchor", False):
            return agent_payload

        proposed_ans = agent_payload.get("final_answer", "")
        if proposed_ans == anchor_ans:
            return agent_payload

        is_valid = await VerifierGate.verify_change(
            proposed_candidate=proposed_ans,
            ground_truth=ground_truth,
            task_domain=task_domain,
            test_cases=test_cases
        )

        if not is_valid:
            reverted = dict(agent_payload)
            reverted["final_answer"] = anchor_ans
            reverted["vote_target"] = anchor_ans
            reverted["changed_from_anchor"] = False
            reverted["revision_evidence"] = (
                f"[VERIFIER GATE REJECTED]: Proposed change '{proposed_ans}' failed deterministic verification. "
                f"Reverted to Round 0 anchor '{anchor_ans}'."
            )
            return reverted

        return agent_payload
