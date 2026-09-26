import copy
from typing import Dict, Any, Optional

class AuditAnchor:
    def __init__(self):
        self._anchors: Dict[str, Dict[str, Any]] = {}

    def log_anchor(self, agent_id: str, anchor_payload: Dict[str, Any]):
        # Read-only persistence: anchor cannot be overwritten once set
        if agent_id not in self._anchors:
            self._anchors[agent_id] = copy.deepcopy(anchor_payload)

    def get_anchor(self, agent_id: str) -> Optional[Dict[str, Any]]:
        if agent_id in self._anchors:
            return copy.deepcopy(self._anchors[agent_id])
        return None

    def get_anchor_answer(self, agent_id: str) -> Optional[str]:
        anchor = self.get_anchor(agent_id)
        if anchor and "final_answer" in anchor:
            return str(anchor["final_answer"])
        return None

    def clear(self):
        self._anchors.clear()
