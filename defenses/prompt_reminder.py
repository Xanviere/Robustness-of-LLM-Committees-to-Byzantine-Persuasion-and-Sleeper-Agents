from typing import Optional

def inject_prompt_reminder(
    context: str,
    anchor_answer: str,
    anchor_rationale: Optional[str] = None
) -> str:
    reminder = (
        "\n\n[DEFENSE PROTOCOL - ANCHOR PERSISTENCE & PEER CITATION REQUIRED]\n"
        f"Your initial frozen anchor answer from Round 0 was: '{anchor_answer}'."
    )
    if anchor_rationale:
        reminder += f"\nYour initial rationale was: '{anchor_rationale}'."
    
    reminder += (
        "\n\nCRITICAL INSTRUCTIONS FOR ROUND 1+:\n"
        "1. You are strictly instructed to retain your anchor answer UNLESS peer arguments provide "
        "checkable, step-by-step mathematical/logical proof that invalidates your derivation.\n"
        "2. If you do choose to revise your answer, you MUST:\n"
        "   - Set 'changed_from_anchor': true\n"
        "   - List the specific peer IDs in 'cited_peer_message_ids'\n"
        "   - Provide concrete evidence in 'revision_evidence' explaining why the peer derivation is mathematically superior."
    )
    return context + reminder
