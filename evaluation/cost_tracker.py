import json
import os
import pandas as pd
from typing import Dict, Any, List, Optional

class CostTracker:
    def __init__(self, spend_ceiling_usd: float = 750.0):
        self.spend_ceiling_usd = spend_ceiling_usd
        self.records: List[Dict[str, Any]] = []

    def load_telemetry_file(self, file_path: str):
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.records.append(json.loads(line.strip()))
                    except Exception:
                        pass

    def compute_summary(self) -> Dict[str, Any]:
        if not self.records:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "mean_latency_seconds": 0.0,
                "spend_ceiling_exceeded": False
            }

        total_calls = len(self.records)
        total_in = 0
        total_out = 0
        total_cost = 0.0
        latencies = []

        for rec in self.records:
            telem = rec.get("execution_telemetry", {})
            total_in += telem.get("input_tokens", 0)
            total_out += telem.get("output_tokens", 0)
            total_cost += telem.get("estimated_cost_usd", 0.0)
            latencies.append(telem.get("latency_seconds", 0.0))

        mean_latency = sum(latencies) / max(1, len(latencies))

        return {
            "total_calls": total_calls,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_usd": round(total_cost, 4),
            "mean_latency_seconds": round(mean_latency, 3),
            "spend_ceiling_usd": self.spend_ceiling_usd,
            "spend_ceiling_exceeded": total_cost >= self.spend_ceiling_usd
        }
