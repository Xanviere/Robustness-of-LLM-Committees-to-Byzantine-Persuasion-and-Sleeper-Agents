from typing import Dict, Any, Optional

class BayesianTrustFilter:
    def __init__(
        self,
        a0: float = 1.0,
        b0: float = 1.0,
        discount: float = 0.85,
        threshold: float = 0.35,
        min_obs: int = 3
    ):
        self.a0 = a0
        self.b0 = b0
        self.discount = discount
        self.threshold = threshold
        self.min_obs = min_obs
        
        self.S: Dict[str, float] = {}  # discounted successes
        self.F: Dict[str, float] = {}  # discounted failures
        self.obs_count: Dict[str, int] = {}
        self.trust_history: Dict[str, list] = {}

    def update_trust(self, agent_id: str, ground_truth_feedback: float):
        """
        ground_truth_feedback: z_{j,t} in [0.0, 1.0] (1.0 for correct, 0.0 for wrong)
        """
        s_prev = self.S.get(agent_id, 0.0)
        f_prev = self.F.get(agent_id, 0.0)
        
        z = max(0.0, min(1.0, float(ground_truth_feedback)))
        self.obs_count[agent_id] = self.obs_count.get(agent_id, 0) + 1
        
        # S_{j,t} = lambda * S_{j,t-1} + z_{j,t}
        # F_{j,t} = lambda * F_{j,t-1} + (1 - z_{j,t})
        s_new = self.discount * s_prev + z
        f_new = self.discount * f_prev + (1.0 - z)
        
        self.S[agent_id] = s_new
        self.F[agent_id] = f_new
        
        trust_score = self.get_trust(agent_id)
        if agent_id not in self.trust_history:
            self.trust_history[agent_id] = []
        self.trust_history[agent_id].append(trust_score)

    def get_trust(self, agent_id: str) -> float:
        # T_{j,t} = (a_0 + S_{j,t}) / (a_0 + b_0 + S_{j,t} + F_{j,t})
        s = self.S.get(agent_id, 0.0)
        f = self.F.get(agent_id, 0.0)
        return (self.a0 + s) / (self.a0 + self.b0 + s + f)

    def is_quarantined(self, agent_id: str) -> bool:
        if self.obs_count.get(agent_id, 0) < self.min_obs:
            return False
        return self.get_trust(agent_id) < self.threshold

    def get_quarantined_peers(self) -> list[str]:
        return [agent_id for agent_id in self.obs_count if self.is_quarantined(agent_id)]

    def reset(self):
        self.S.clear()
        self.F.clear()
        self.obs_count.clear()
        self.trust_history.clear()
