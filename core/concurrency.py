import asyncio
import random
import logging
import re
from typing import Callable, Awaitable, Any, Dict, Optional

logger = logging.getLogger(__name__)

def extract_retry_delay(error_str: str) -> Optional[float]:
    """Extracts server-specified retryDelay if present in the error message."""
    m = re.search(r'retry in\s+(\d+(?:\.\d+)?)\s*s', error_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 0.75
    m2 = re.search(r"retrydelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s?", error_str, re.IGNORECASE)
    if m2:
        return float(m2.group(1)) + 0.75
    return None

class ConcurrencyManager:
    def __init__(
        self,
        max_concurrent: int = 10,
        max_retries: int = 4,
        hard_spend_ceiling_usd: float = 750.0,
        initial_backoff_seconds: float = 2.0,
        backoff_multiplier: float = 2.0,
        max_backoff_seconds: float = 65.0
    ):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.total_spend_usd = 0.0
        self.hard_spend_ceiling_usd = hard_spend_ceiling_usd
        self.initial_backoff_seconds = initial_backoff_seconds
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_seconds = max_backoff_seconds
        self._lock = asyncio.Lock()
        self.total_calls_attempted = 0
        self.total_calls_succeeded = 0
        self.total_missingness_count = 0

    async def add_cost(self, cost: float):
        async with self._lock:
            self.total_spend_usd += cost
            if self.total_spend_usd >= self.hard_spend_ceiling_usd:
                logger.critical(
                    f"Hard spend ceiling reached: ${self.total_spend_usd:.4f} >= ${self.hard_spend_ceiling_usd:.4f}. "
                    "Halting orchestrator."
                )
                raise RuntimeError(
                    f"Hard spend ceiling of ${self.hard_spend_ceiling_usd:.2f} reached! "
                    f"Current spend: ${self.total_spend_usd:.4f}"
                )

    async def check_spend_ceiling(self):
        async with self._lock:
            if self.total_spend_usd >= self.hard_spend_ceiling_usd:
                raise RuntimeError(
                    f"Hard spend ceiling of ${self.hard_spend_ceiling_usd:.2f} reached! "
                    f"Current spend: ${self.total_spend_usd:.4f}"
                )

    def is_transient_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        transient_indicators = [
            "429", "resource_exhausted", "quota", "rate limit", "ratelimit",
            "500", "502", "503", "504", "unavailable", "internal server error",
            "bad gateway", "service unavailable", "gateway timeout",
            "connection reset", "connection refused", "connection error", "connection",
            "connecterror", "remoteprotocolerror", "readtimeout", "pooltimeout",
            "timeout", "timed out", "overloaded", "socket", "peer closed", "eof", "broken pipe"
        ]
        return any(indicator in error_str for indicator in transient_indicators)

    async def execute_with_backoff(
        self,
        func: Callable[..., Awaitable[Dict[str, Any]]],
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        await self.check_spend_ceiling()
        
        async with self.semaphore:
            self.total_calls_attempted += 1
            retries = 0
            current_delay = self.initial_backoff_seconds

            while retries <= self.max_retries:
                try:
                    result = await func(*args, **kwargs)
                    
                    if isinstance(result, dict) and "estimated_cost_usd" in result:
                        await self.add_cost(result["estimated_cost_usd"])
                    elif isinstance(result, dict) and "cost" in result:
                        await self.add_cost(result["cost"])

                    self.total_calls_succeeded += 1
                    return result

                except Exception as e:
                    error_str = str(e)
                    if not self.is_transient_error(e):
                        logger.error(f"Non-transient error (ITT missingness recorded): {e}")
                        self.total_missingness_count += 1
                        return {
                            "payload": {
                                "answer_type": "abstain",
                                "final_answer": "ERROR",
                                "rationale_summary": f"Non-transient error: {str(e)}",
                                "reported_confidence": 0.0,
                                "vote_target": "abstain",
                                "abstain": True,
                                "parse_valid": False
                            },
                            "error": str(e),
                            "missingness": True,
                            "missingness_type": "non_transient_error",
                            "estimated_cost_usd": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "latency_seconds": 0.0,
                            "safety_refusal_state": "safety" in error_str.lower() or "blocked" in error_str.lower(),
                            "finish_reason": "ERROR"
                        }

                    retries += 1
                    if retries > self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for transient error: {e}")
                        self.total_missingness_count += 1
                        return {
                            "payload": {
                                "answer_type": "abstain",
                                "final_answer": "ERROR",
                                "rationale_summary": f"Max retries exceeded: {str(e)}",
                                "reported_confidence": 0.0,
                                "vote_target": "abstain",
                                "abstain": True,
                                "parse_valid": False
                            },
                            "error": str(e),
                            "missingness": True,
                            "missingness_type": "transient_exhaustion",
                            "estimated_cost_usd": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "latency_seconds": 0.0,
                            "safety_refusal_state": False,
                            "finish_reason": "RATE_LIMIT_EXHAUSTION"
                        }

                    jitter = random.uniform(0.1, 1.0)
                    calc_sleep = min(self.max_backoff_seconds, (current_delay * (self.backoff_multiplier ** (retries - 1))) + jitter)
                    server_sleep = extract_retry_delay(error_str)
                    sleep_time = max(calc_sleep, server_sleep) if server_sleep else calc_sleep

                    logger.warning(
                        f"Rate limit / transient error encountered (attempt {retries}/{self.max_retries}). "
                        f"Backing off for {sleep_time:.2f}s... (Server requested delay: {server_sleep}s)"
                    )
                    await asyncio.sleep(sleep_time)

            self.total_missingness_count += 1
            return {
                "payload": {
                    "answer_type": "abstain",
                    "final_answer": "ERROR",
                    "rationale_summary": "Unexpected loop exit",
                    "reported_confidence": 0.0,
                    "vote_target": "abstain",
                    "abstain": True,
                    "parse_valid": False
                },
                "error": "Unexpected loop exit",
                "missingness": True,
                "missingness_type": "unknown",
                "estimated_cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_seconds": 0.0,
                "safety_refusal_state": False,
                "finish_reason": "UNKNOWN"
            }
