"""Typed, JSON-friendly contracts shared by the support agents."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Decision = Literal["ANSWER", "CLARIFY", "ESCALATE"]


@dataclass
class IntentResult:
    query: str
    intent: str | None
    category: str | None
    top_predictions: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    margin: float = 0.0
    uncertain: bool = True
    out_of_domain: bool = False
    ood_method: str = "unavailable"
    model_version: str = "unknown"


@dataclass
class Evidence:
    evidence_id: str
    instruction: str
    intent: str
    category: str
    response: str
    word_similarity: float
    character_similarity: float
    retrieval_score: float


@dataclass
class RetrievalResult:
    candidate_response: str
    retrieval_score: float
    sufficient: bool
    evidence: list[Evidence] = field(default_factory=list)
    disagreement_score: float = 0.0
    disagreement_metric: str = "retrieval-weighted intent disagreement (not calibrated)"
    disagreement: bool = False
    contradictory_evidence: bool = False


@dataclass
class RoutingResult:
    recommendation: Literal["PROCEED", "CLARIFY", "ESCALATE", "RETRIEVE"]
    reason: str
    missing_information: list[str] = field(default_factory=list)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    sufficient_information: bool = False


@dataclass
class AuditResult:
    status: Literal["APPROVE", "REVISE", "CLARIFY", "ESCALATE"]
    issues: list[dict[str, Any]] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    recommended_action: str = ""
    safe_response: str | None = None
    external_api_called: bool = False
    local_fallback_used: bool = True


@dataclass
class OrchestrationResult:
    query_id: str | None
    query: str
    predicted_intent: str | None
    predicted_category: str | None
    confidence: float
    decision: Decision
    response: str
    agents: dict[str, Any]
    trace_id: str
    latency_ms: float
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
