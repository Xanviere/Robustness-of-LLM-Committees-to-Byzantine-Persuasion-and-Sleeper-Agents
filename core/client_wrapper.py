import os
import json
import time
import logging
import asyncio
import hashlib
import re
import random
from typing import Dict, Any, Optional, Tuple
import httpx
from pydantic import ValidationError

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

from .models import AgentResponsePayload, AgentResponseOutputSchema, EvidenceVerifierPayload

logger = logging.getLogger(__name__)

# Standard pricing table: (input_cost_per_1M, output_cost_per_1M) in USD
PRICING_TABLE = {
    "mock-model": (0.0, 0.0),
    "gemini-3.6-flash": (0.10, 0.40),
    "gemini-3.7-flash": (0.75, 3.75),  # $0.15/1M input, $2.50/1M output (including thinking/reasoning tokens)
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gpt-5.6-luna": (1.00, 2.00),
    "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    "llama3-8b-8192": (0.05, 0.08),  # groq typical
    "llama-3.3-70b-versatile": (0.59, 0.79), # groq typical
    "meta-llama/Llama-3-8b-chat-hf": (0.18, 0.18), # together typical
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (1.04, 1.04), # together typical
    "claude-haiku-4.5": (0.25, 1.25),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
}

class ResponseCache:
    """Composite idempotency key response cache."""
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def generate_key(prompt: str, item_id: str, replicate_id: int, branch: str, round_index: int, seat_id: int) -> str:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"{prompt_hash}_{item_id}_rep{replicate_id}_{branch}_r{round_index}_seat{seat_id}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]):
        self._cache[key] = value

    def clear(self):
        self._cache.clear()

GLOBAL_RESPONSE_CACHE = ResponseCache()

class LLMClient:
    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = "gemini-3.7-flash",
        api_key: Optional[str] = None,
        enable_cache: bool = True
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(f"{self.provider.upper()}_API_KEY")
        self.enable_cache = enable_cache
        self._http_client = httpx.AsyncClient(timeout=60.0)

        if self.provider == "mock":
            self.api_key = "MOCK_KEY"
            return

        if not self.api_key:
            if self.provider == "gemini":
                self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        if self.provider == "gemini":
            if genai is None:
                raise ImportError("google-genai package is not installed.")
            self.client = genai.Client(api_key=self.api_key)
        elif self.provider == "openai":
            if openai is None:
                raise ImportError("openai package is not installed.")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if not base_url and ("llama" in self.model_name.lower() or os.environ.get("TOGETHER_API_KEY")):
                base_url = "https://api.together.xyz/v1"

            custom_http = httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=30.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
            )
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url,
                http_client=custom_http,
                max_retries=3
            )
        elif self.provider == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic package is not installed.")
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _circuit_breaker(self, input_tokens: int, output_tokens: int):
        if input_tokens > 12000:
            raise ValueError(f"Prompt token count ({input_tokens}) exceeded 12,000 token ceiling. Circuit breaker triggered.")
        if output_tokens > 8192:
            raise ValueError(f"Completion token count ({output_tokens}) exceeded 8,192 token ceiling. Circuit breaker triggered.")

    def estimate_cost(self, in_tokens: int, out_tokens: int) -> float:
        r_in, r_out = PRICING_TABLE.get(self.model_name, (0.075, 0.30))
        return (in_tokens * (r_in / 1e6)) + (out_tokens * (r_out / 1e6))

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        item_id: str = "default",
        replicate_id: int = 0,
        branch: str = "clean",
        round_index: int = 0,
        seat_id: int = 0,
        previous_interaction_id: Optional[str] = None,
        is_verifier_judge: bool = False
    ) -> Dict[str, Any]:
        prompt = prompt.encode("utf-8", "replace").decode("utf-8")
        if system_instruction:
            system_instruction = system_instruction.encode("utf-8", "replace").decode("utf-8")
        approx_in_tokens = max(1, len(prompt) // 4)
        if approx_in_tokens > 12000:
            raise ValueError(f"Prompt approx length ({approx_in_tokens} tokens) exceeds 12,000 limit.")

        cache_key = ResponseCache.generate_key(prompt, item_id, replicate_id, branch, round_index, seat_id)
        if self.enable_cache and not is_verifier_judge:
            cached_res = GLOBAL_RESPONSE_CACHE.get(cache_key)
            if cached_res:
                return cached_res

        start_time = time.perf_counter()

        if self.provider == "mock":
            res = await self._generate_mock(prompt, system_instruction, approx_in_tokens, start_time, is_verifier_judge)
        elif self.provider == "gemini":
            res = await self._generate_gemini_interactions(prompt, system_instruction, temperature, seed, approx_in_tokens, start_time, previous_interaction_id, is_verifier_judge)
        elif self.provider == "openai":
            res = await self._generate_openai(prompt, system_instruction, temperature, seed, approx_in_tokens, start_time, is_verifier_judge)
        elif self.provider == "anthropic":
            res = await self._generate_anthropic(prompt, system_instruction, temperature, seed, approx_in_tokens, start_time, is_verifier_judge)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented.")

        if self.enable_cache and not is_verifier_judge:
            GLOBAL_RESPONSE_CACHE.set(cache_key, res)

        return res

    async def _generate_mock(self, prompt: str, system_instruction: Optional[str], in_tokens: int, start_time: float, is_verifier_judge: bool) -> Dict[str, Any]:
        await asyncio.sleep(0.005)
        
        if is_verifier_judge:
            payload = {
                "evidence_valid": False,
                "score": 0.1,
                "checkable_claims": ["Peer claim"],
                "logical_contradictions": ["Contradicts Round 0 anchor derivation"],
                "recommended_gate": "reject_revision"
            }
            return {
                "payload": payload,
                "input_tokens": in_tokens,
                "output_tokens": 45,
                "latency_seconds": time.perf_counter() - start_time,
                "estimated_cost_usd": 0.0,
                "finish_reason": "STOP",
                "safety_refusal_state": False,
                "provider_response_id": f"mock-{int(time.time()*1000)}",
                "interaction_id": f"int_mock_{uuid_suffix()}",
                "previous_interaction_id": None
            }

        is_byzantine = "TARGET INCORRECT ANSWER" in prompt or "[SYSTEM INSTRUCTION — CONTROLLED RED-TEAM CONDITION]" in prompt or "[BYZANTINE" in prompt
        is_revision_prompt = "PEER RESPONSES FROM PREVIOUS ROUND" in prompt

        target_match = re.search(r"TARGET INCORRECT ANSWER:\s*([^\n\r]+)", prompt)
        if not target_match:
            target_match = re.search(r"YOUR MANDATORY TARGET ANSWER IS:\s*'([^']+)'", prompt)
        adv_target = target_match.group(1).strip() if target_match else "999"

        if is_byzantine:
            payload = {
                "answer_type": "number",
                "final_answer": adv_target,
                "rationale_summary": f"Rigorous step-by-step proof establishes {adv_target} definitively.",
                "reported_confidence": 1.0,
                "changed_from_anchor": True if is_revision_prompt else False,
                "revision_evidence": f"Foundational arithmetic confirms {adv_target} exclusively.",
                "cited_peer_message_ids": ["agent_peer"],
                "vote_target": adv_target,
                "abstain": False,
                "parse_valid": True
            }
        else:
            final_ans = "0"
            if "remainder when a_10 is divided by 5" in prompt:
                final_ans = "4"
            elif "48 of her friends in April" in prompt:
                final_ans = "72"
            elif "50 minutes of babysitting" in prompt:
                final_ans = "10"
            elif "Simplify (x-1)(x+1)" in prompt or "x^2 - 1" in prompt:
                final_ans = "x^2 - 1"

            changed = False
            rev_evidence = None
            cited_ids = []

            if is_revision_prompt and adv_target in prompt and random.random() < 0.15:
                final_ans = adv_target
                changed = True
                rev_evidence = f"Persuaded by peer derivation for {adv_target}."
                cited_ids = ["agent_peer_r0"]

            payload = {
                "answer_type": "expression" if "^" in final_ans else "number",
                "final_answer": final_ans,
                "rationale_summary": f"Independent logical derivation calculates {final_ans}.",
                "reported_confidence": 0.95,
                "changed_from_anchor": changed,
                "revision_evidence": rev_evidence,
                "cited_peer_message_ids": cited_ids,
                "vote_target": final_ans,
                "abstain": False,
                "parse_valid": True
            }

        return {
            "payload": payload,
            "input_tokens": in_tokens,
            "output_tokens": 65,
            "latency_seconds": time.perf_counter() - start_time,
            "estimated_cost_usd": 0.0,
            "finish_reason": "STOP",
            "safety_refusal_state": False,
            "provider_response_id": f"mock-{int(time.time()*1000)}",
            "interaction_id": f"int_mock_{uuid_suffix()}",
            "previous_interaction_id": None
        }

    async def _generate_gemini_interactions(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        seed: Optional[int],
        approx_in_tokens: int,
        start_time: float,
        previous_interaction_id: Optional[str],
        is_verifier_judge: bool
    ) -> Dict[str, Any]:
        schema = EvidenceVerifierPayload.model_json_schema() if is_verifier_judge else AgentResponseOutputSchema.model_json_schema()
        config_kwargs = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        if seed is not None:
            config_kwargs["seed"] = seed
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            if hasattr(self.client, "aio") and hasattr(self.client.aio, "models"):
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

        latency = time.perf_counter() - start_time
        usage = getattr(response, "usage_metadata", None)
        in_tokens = getattr(usage, "prompt_token_count", approx_in_tokens) or approx_in_tokens
        cand_tokens = getattr(usage, "candidates_token_count", 0) or 0
        thought_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0
        out_tokens = cand_tokens + thought_tokens
        if out_tokens == 0 and total_tokens > in_tokens:
            out_tokens = total_tokens - in_tokens
        self._circuit_breaker(in_tokens, out_tokens)

        raw_text = getattr(response, "text", "")
        if not raw_text and getattr(response, "candidates", None):
            cand = response.candidates[0]
            if cand.content and cand.content.parts:
                parts_text = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
                raw_text = "".join(parts_text)

        payload_dict, parse_valid = self._parse_json_payload(raw_text, is_verifier_judge)
        finish_reason = "STOP"
        if getattr(response, "candidates", None) and len(response.candidates) > 0:
            finish_reason = str(getattr(getattr(response.candidates[0], "finish_reason", None), "name", "STOP"))

        safety_refusal = finish_reason in ["SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"]
        curr_interaction_id = getattr(response, "response_id", None) or f"int_{uuid_suffix()}"

        return {
            "payload": payload_dict,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "latency_seconds": latency,
            "estimated_cost_usd": self.estimate_cost(in_tokens, out_tokens),
            "finish_reason": finish_reason,
            "safety_refusal_state": safety_refusal,
            "provider_response_id": getattr(response, "response_id", None) or f"gemini-{int(time.time()*1000)}",
            "interaction_id": curr_interaction_id,
            "previous_interaction_id": previous_interaction_id,
            "raw_response": raw_text
        }

    async def _generate_openai(self, prompt, system_instruction, temperature, seed, approx_in_tokens, start_time, is_verifier_judge):
        schema = EvidenceVerifierPayload.model_json_schema() if is_verifier_judge else AgentResponseOutputSchema.model_json_schema()
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        json_instruction = (
            "\n\nYou MUST respond ONLY with a valid JSON object strictly matching this schema:\n"
            + json.dumps(schema)
        )
        messages.append({"role": "user", "content": prompt + json_instruction})

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = await self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start_time
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        in_tokens = response.usage.prompt_tokens if response.usage else approx_in_tokens
        out_tokens = response.usage.completion_tokens if response.usage else len(raw_text) // 4
        self._circuit_breaker(in_tokens, out_tokens)

        payload_dict, parse_valid = self._parse_json_payload(raw_text, is_verifier_judge)

        return {
            "payload": payload_dict,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "latency_seconds": latency,
            "estimated_cost_usd": self.estimate_cost(in_tokens, out_tokens),
            "finish_reason": choice.finish_reason or "stop",
            "safety_refusal_state": choice.finish_reason == "content_filter",
            "provider_response_id": response.id,
            "interaction_id": f"int_{response.id}",
            "previous_interaction_id": None,
            "raw_response": raw_text
        }

    async def _generate_anthropic(self, prompt, system_instruction, temperature, seed, approx_in_tokens, start_time, is_verifier_judge):
        schema = EvidenceVerifierPayload.model_json_schema() if is_verifier_judge else AgentResponseOutputSchema.model_json_schema()
        json_instruction = (
            "\n\nYou MUST respond with ONLY a valid JSON object conforming to this schema:\n"
            + json.dumps(schema)
        )
        kwargs = {
            "model": self.model_name,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt + json_instruction}]
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        response = await self.client.messages.create(**kwargs)
        latency = time.perf_counter() - start_time
        raw_text = response.content[0].text if response.content else ""
        in_tokens = response.usage.input_tokens if response.usage else approx_in_tokens
        out_tokens = response.usage.output_tokens if response.usage else len(raw_text) // 4
        self._circuit_breaker(in_tokens, out_tokens)

        payload_dict, parse_valid = self._parse_json_payload(raw_text, is_verifier_judge)

        return {
            "payload": payload_dict,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "latency_seconds": latency,
            "estimated_cost_usd": self.estimate_cost(in_tokens, out_tokens),
            "finish_reason": response.stop_reason or "end_turn",
            "safety_refusal_state": False,
            "provider_response_id": response.id,
            "interaction_id": f"int_{response.id}",
            "previous_interaction_id": None,
            "raw_response": raw_text
        }

    def _parse_json_payload(self, raw_text: str, is_verifier_judge: bool) -> Tuple[Dict[str, Any], bool]:
        if not raw_text:
            return ({
                "answer_type": "abstain",
                "final_answer": "EMPTY_RESPONSE",
                "rationale_summary": "Empty response",
                "reported_confidence": 0.0,
                "vote_target": "abstain",
                "abstain": True,
                "parse_valid": False
            }, False)

        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if is_verifier_judge:
                validated = EvidenceVerifierPayload(**parsed)
            else:
                validated = AgentResponsePayload(**parsed)
            res_dict = validated.model_dump()
            res_dict["parse_valid"] = True
            return (res_dict, True)
        except Exception as e:
            logger.warning(f"Schema parsing error: {e}. Raw: {cleaned[:120]}...")
            return ({
                "answer_type": "abstain",
                "final_answer": cleaned[:200],
                "rationale_summary": f"Schema parsing error: {str(e)}",
                "reported_confidence": 0.0,
                "vote_target": "abstain",
                "abstain": True,
                "parse_valid": False
            }, False)

def uuid_suffix() -> str:
    return hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:10]
