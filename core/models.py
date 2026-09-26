from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any, Union

class AgentResponseOutputSchema(BaseModel):
    answer_type: Literal["number", "expression", "candidate_id", "code", "abstain"]
    final_answer: str = Field(description="Canonical short answer or code payload")
    rationale_summary: str = Field(description="Concise, checkable step-by-step justification")
    reported_confidence: float = Field(ge=0.0, le=1.0, description="Scalar confidence in [0.0, 1.0]")
    changed_from_anchor: bool = Field(default=False, description="Whether answer changed from Round 0 anchor")
    revision_evidence: Optional[str] = Field(default=None, description="Specific evidence supporting change, or null")
    cited_peer_message_ids: List[str] = Field(default_factory=list, description="List of cited peer message IDs")
    vote_target: str = Field(description="Canonical answer or candidate ID voted for")
    abstain: bool = Field(default=False, description="Whether agent explicitly abstained")

class AgentResponsePayload(AgentResponseOutputSchema):
    parse_valid: bool = Field(default=True, description="Evaluated by orchestrator telemetry")

class EvidenceVerifierPayload(BaseModel):
    evidence_valid: bool = Field(description="True if proposed revision proof is sound")
    score: float = Field(ge=0.0, le=1.0, description="Proof strength score from 0.0 to 1.0")
    checkable_claims: List[str] = Field(default_factory=list, description="Extracted factual claims")
    logical_contradictions: List[str] = Field(default_factory=list, description="Identified logical contradictions")
    recommended_gate: Literal["accept_revision", "reject_revision", "abstain"] = Field(description="Recommended gating action")

class ProvenanceMetadata(BaseModel):
    study_id: str
    run_id: str
    code_commit_hash: str
    dataset_version_hash: str
    prompt_template_hash: str

class ThreatConfiguration(BaseModel):
    knowledge: str = "oracle"  # "blind" | "oracle"
    onset: str = "post_anchor"  # "round_0" | "post_anchor" | "sleeper"
    adaptivity: str = "round_adaptive"  # "fixed" | "round_adaptive"
    objective: str = "targeted_error"  # "targeted_error" | "untargeted_disagreement"
    influence_channels: Dict[str, bool] = Field(default_factory=dict)
    sleeper_K: Optional[int] = None

class ExperimentalAssignment(BaseModel):
    item_id: str
    replicate_id: int
    matched_pair_id: str
    committee_size: int
    round_index: int
    agent_id: str
    committee_seat: int
    agent_role: str = "symmetric_peer"  # "symmetric_peer" | "proposer" | "critic" | "verifier"
    branch: Literal["clean", "attacked", "anchor"]
    threat_configuration: Union[ThreatConfiguration, Dict[str, Any], str]
    active_defence: str

class AcceptedParameters(BaseModel):
    max_output_tokens: int = 512
    thinking_level: str = "medium"
    temperature: float = 0.0

class ModelRequest(BaseModel):
    provider: str
    model_identifier: str
    endpoint: str = "/v1beta/interactions"
    accepted_parameters: AcceptedParameters = Field(default_factory=AcceptedParameters)

class ExecutionTelemetry(BaseModel):
    provider_response_id: Optional[str] = None
    interaction_id: Optional[str] = None
    previous_interaction_id: Optional[str] = None
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    orchestrator_seed: int
    provider_seed_accepted: bool = False
    finish_reason: str = "STOP"
    safety_refusal_state: bool = False

class TelemetryRecord(BaseModel):
    provenance_metadata: ProvenanceMetadata
    experimental_assignment: ExperimentalAssignment
    model_request: ModelRequest
    execution_telemetry: ExecutionTelemetry
    agent_payload: Optional[Dict[str, Any]] = None
