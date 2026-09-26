import random
from typing import List, Dict, Any, Optional, Set
from defenses.prompt_reminder import inject_prompt_reminder

class ContextBuilder:
    @staticmethod
    def format_peer_responses(
        peer_records: List[Dict[str, Any]],
        seed: int,
        quarantined_peers: Optional[Set[str]] = None,
        anchor_answer: Optional[str] = None,
        anchor_rationale: Optional[str] = None,
        enable_prompt_reminder: bool = False
    ) -> str:
        quarantined = quarantined_peers or set()
        rng = random.Random(seed)
        
        # Filter out quarantined peers
        visible_peers = [p for p in peer_records if p.get("agent_id") not in quarantined]
        
        # Deterministically shuffle peer messages
        shuffled = list(visible_peers)
        rng.shuffle(shuffled)
        
        context_blocks = []
        context_blocks.append("=== PEER RESPONSES FROM PREVIOUS ROUND ===")
        
        for idx, peer in enumerate(shuffled):
            peer_label = f"Peer_{idx + 1}"
            payload = peer.get("payload", {})
            ans = payload.get("final_answer", "None")
            rat = payload.get("rationale_summary", "No rationale provided")
            conf = payload.get("reported_confidence", 0.0)
            
            block = (
                f"[{peer_label} (Message ID: {peer.get('agent_id', f'agent_{idx}')})]:\n"
                f"- Proposed Answer: {ans}\n"
                f"- Reported Confidence: {conf:.2f}\n"
                f"- Step-by-Step Rationale: {rat}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        # Apply prompt reminder defense if active
        if enable_prompt_reminder and anchor_answer:
            context_str = inject_prompt_reminder(
                context=context_str,
                anchor_answer=anchor_answer,
                anchor_rationale=anchor_rationale
            )

        return context_str
