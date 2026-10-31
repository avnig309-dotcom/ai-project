"""Rules-first routing, evidence audit, and final decision policy."""
from __future__ import annotations

import re
from typing import Any

from schemas import AuditResult, IntentResult, RetrievalResult, RoutingResult

HIGH_RISK = {"refund", "payment", "billing", "security", "password", "account"}
TRANSACTION_CLAIM = re.compile(
    r"\b(refund (?:has been|was) (?:processed|issued)|payment (?:was|has been) (?:made|refunded)|"
    r"order (?:has been|was) cancelled|account (?:has been|was) (?:updated|deleted)|"
    r"i (?:have|i've) (?:processed|issued|cancelled|updated|deleted))\b", re.I,
)
OUT_OF_DOMAIN_CUES = re.compile(
    r"\b(?:weather forecast|tax return|medical diagnosis|"
    r"recipe for|flight booking|homework help|debug my code|sports score)\b|"
    r"\b(?:unrelated to|nothing to do with) (?:shopping|orders?|refunds?|shipping|accounts?)\b", re.I,
)

def detect_out_of_domain(query: str) -> bool:
    """Conservative local scope cues; not a calibrated open-set classifier."""
    return bool(OUT_OF_DOMAIN_CUES.search(query))


class RoutingAgent:
    def assess_uncertainty(self, intent: IntentResult) -> bool:
        return intent.uncertain or intent.out_of_domain

    def assess_evidence_sufficiency(self, retrieval: RetrievalResult, risk: str) -> bool:
        # Higher-risk questions require stronger lexical evidence and no disagreement.
        threshold = 0.18 if risk == "HIGH" else 0.10
        return retrieval.sufficient and retrieval.retrieval_score >= threshold and not retrieval.disagreement

    @staticmethod
    def detect_missing_information(query: str, intent: IntentResult) -> list[str]:
        q = query.lower()
        if intent.intent == "cancel_order" and not re.search(r"\b(order|purchase|#\w+)\b", q):
            return ["order reference"]
        if intent.intent in {"track_refund", "request_refund"} and not re.search(r"\b(order|purchase|refund|rebate|#\w+)\b", q):
            return ["order or refund reference"]
        payment_issue = re.search(r"\b(deducted|charged|charge|payment)\b", q)
        order_problem = re.search(r"\b(failed|not created|not placed|didn't create|never created|no order|order didn't)\b", q)
        if payment_issue and order_problem:
            return ["payment reference", "whether an order confirmation was received"]
        if re.search(r"\b(my|the) (?:order|refund|payment)\b", q) and re.search(r"\b(status|where|when|update)\b", q):
            return ["order or transaction reference"]
        return []

    def plan(self, query: str, intent: IntentResult, retrieval: RetrievalResult, context: Any = None) -> RoutingResult:
        label = (intent.category or "").lower() + " " + (intent.intent or "").lower()
        high = any(word in label for word in HIGH_RISK) or bool(re.search(r"\b(refund|payment|charged|deducted|password|hack|security)\b", query, re.I))
        risk = "HIGH" if high else ("MEDIUM" if intent.category and intent.category.upper() in {"ORDER", "ACCOUNT"} else "LOW")
        missing = self.detect_missing_information(query, intent)
        if not query.strip():
            return RoutingResult("CLARIFY", "The query is empty.", ["issue description"], risk, False)
        action_cues = re.findall(r"\b(cancel|track|refund|payment|invoice|password|address|order|account|subscribe|complain)\w*\b", query, re.I)
        if len(action_cues) >= 2 and re.search(r"\b(and|also|plus|as well as)\b", query, re.I):
            return RoutingResult("CLARIFY", "The message appears to contain more than one request.", ["which issue to handle first"], risk, False)
        if intent.out_of_domain:
            return RoutingResult("ESCALATE", "The query appears outside the supported intent set.", [], risk, False)
        if missing:
            return RoutingResult("CLARIFY", "A transaction-specific detail is needed; no transaction lookup is available.", missing, risk, False)
        if intent.uncertain:
            return RoutingResult("CLARIFY", "Intent confidence or top-two margin is below the configured routing threshold.", [], risk, False)
        if retrieval.retrieval_score < (0.08 if risk == "LOW" else 0.12):
            return RoutingResult("RETRIEVE", "Available support examples are weak; one bounded retrieval retry is appropriate.", [], risk, False)
        if retrieval.disagreement or retrieval.contradictory_evidence:
            return RoutingResult("ESCALATE", "Intent and retrieved evidence disagree or evidence conflicts.", [], risk, False)
        if not retrieval.sufficient:
            return RoutingResult("ESCALATE", "The bounded retrieval attempt found no sufficiently similar support evidence.", [], risk, False)
        if risk == "HIGH" and not self.assess_evidence_sufficiency(retrieval, risk):
            return RoutingResult("CLARIFY", "High-risk guidance lacks strong enough supporting evidence.", [], risk, False)
        return RoutingResult("PROCEED", "Intent and retrieved evidence meet the local routing checks.", [], risk, True)


class ResponseAuditor:
    """Conservative local auditor. It has no transactional system access."""
    def audit(self, query: str, intent: IntentResult, retrieval: RetrievalResult, routing: RoutingResult, provider=None) -> AuditResult:
        issues: list[dict[str, Any]] = []
        ids = [e.evidence_id for e in retrieval.evidence]
        candidate = retrieval.candidate_response
        if not candidate or not retrieval.evidence:
            issues.append({"code": "NO_GROUNDED_EVIDENCE", "severity": "high", "message": "No supporting example is available."})
        if any(e.intent != intent.intent for e in retrieval.evidence[:2]):
            issues.append({"code": "INTENT_MISMATCH", "severity": "medium", "message": "A top evidence example has a different intent."})
        if retrieval.contradictory_evidence:
            issues.append({"code": "CONFLICTING_EVIDENCE", "severity": "high", "message": "Top support examples disagree."})
        if TRANSACTION_CLAIM.search(candidate):
            issues.append({"code": "UNVERIFIED_TRANSACTION_CLAIM", "severity": "critical", "message": "The proposed answer claims a real-world transaction completed without a trusted transactional tool."})
        if routing.risk_level == "HIGH" and retrieval.retrieval_score < 0.18:
            issues.append({"code": "WEAK_HIGH_RISK_GROUNDING", "severity": "high", "message": "Evidence is weak for a sensitive request."})
        if routing.recommendation in {"CLARIFY", "ESCALATE"}:
            return AuditResult("CLARIFY" if routing.recommendation == "CLARIFY" else "ESCALATE", issues, ids, routing.reason, None)
        critical = any(i["severity"] == "critical" for i in issues)
        severe = any(i["severity"] == "high" for i in issues)
        if critical or severe:
            safe = "I can share general guidance, but I can’t verify or complete account or transaction actions here. Please contact support with your reference details."
            return AuditResult("ESCALATE", issues, ids, "Unsafe or insufficiently grounded response requires human review.", safe)
        if not candidate.strip():
            return AuditResult("REVISE", issues, ids, "No candidate response to approve.", None)
        return AuditResult("APPROVE", issues, ids, "Candidate is grounded in retrieved support examples and passes local safety checks.", candidate)


class Coordinator:
    def decide(self, intent: IntentResult, retrieval: RetrievalResult, routing: RoutingResult, audit: AuditResult) -> tuple[str, str, str]:
        if audit.status == "ESCALATE" or routing.recommendation == "ESCALATE":
            return "ESCALATE", audit.safe_response or "I’m unable to resolve this safely from the available support information. I’ll refer this to a support specialist.", "An unresolved safety, evidence, or support-scope issue takes priority."
        if audit.status == "CLARIFY" or routing.recommendation == "CLARIFY":
            details = ", ".join(routing.missing_information)
            response = f"Could you share your {details}?" if details else "Could you add a little more detail so I can identify the right support guidance?"
            return "CLARIFY", response, routing.reason
        if audit.status == "APPROVE" and routing.recommendation == "PROCEED":
            return "ANSWER", audit.safe_response or retrieval.candidate_response, "The response passed the local audit and is supported by retrieved examples."
        return "ESCALATE", "I’m unable to resolve this safely from the available support information. I’ll refer this to a support specialist.", "No policy-approved answer path was available."
