import copy
from typing import Dict, Any, Tuple

class BranchCloner:
    @staticmethod
    def clone_anchor(anchor_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Takes the frozen Master Anchor dictionary from Round 0
        and creates two completely independent, deep-copied branches:
        Clean Branch and Attacked Branch.
        """
        clean_branch = copy.deepcopy(anchor_state)
        attacked_branch = copy.deepcopy(anchor_state)
        return clean_branch, attacked_branch
