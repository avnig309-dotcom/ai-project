"""Five-agent pipeline over the existing trained two-agent joblib bundle."""
from __future__ import annotations

import hashlib
import time
import unicodedata
import uuid
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from schemas import Evidence, IntentResult, OrchestrationResult, RetrievalResult
from support_agents import Coordinator, ResponseAuditor, RoutingAgent, detect_out_of_domain


class FiveAgentSystem:
    def __init__(self, two_agent_system, model_version: str | None = None, debate_enabled: bool = True):
        self.legacy = two_agent_system
        self.agent1 = two_agent_system.agent1
        self.agent2 = two_agent_system.agent2
        self.agent3 = RoutingAgent()
        self.agent4 = ResponseAuditor()
        self.agent5 = Coordinator()
        self.model_version = model_version or getattr(two_agent_system, "model_version", "legacy-joblib")
        self.debate_enabled = debate_enabled

    @classmethod
    def from_joblib(cls, path: str):
        from agents_pipeline import load_system
        return cls(load_system(path))

    def _analyze(self, query: str, probabilities=None) -> IntentResult:
        if not query.strip():
            return IntentResult(query, None, None)
        query = unicodedata.normalize("NFKC", query).strip()
        model = self.agent1.model
        probabilities = probabilities if probabilities is not None else model.predict_proba([query])[0]
        labels = model.named_steps["classifier"].classes_
        order = np.argsort(probabilities)[::-1]
        top = [{"intent": str(labels[i]), "probability": round(float(probabilities[i]), 6)} for i in order[:3]]
        confidence = float(probabilities[order[0]])
        margin = confidence - (float(probabilities[order[1]]) if len(order) > 1 else 0.0)
        min_confidence, min_margin = getattr(self.agent1, "uncertainty_thresholds", (0.05, 0.005))
        return IntentResult(
            query=query, intent=str(labels[order[0]]),
            category=self.agent1.intent_to_category.get(str(labels[order[0]]), "UNKNOWN"),
            top_predictions=top, confidence=confidence, margin=margin,
            uncertain=bool(confidence < min_confidence or margin < min_margin), out_of_domain=detect_out_of_domain(query),
            ood_method="conservative local scope cues; not calibrated (no permitted OOD-labeled validation set supplied)",
            model_version=self.model_version,
        )

    def _retrieve(self, query: str, intent: IntentResult, top_k: int = 3) -> RetrievalResult:
        a2 = self.agent2
        if not query:
            return RetrievalResult("", 0.0, False)
        w = cosine_similarity(a2.word_vectorizer.transform([query]), a2.word_vectors).ravel()
        c = cosine_similarity(a2.char_vectorizer.transform([query]), a2.char_vectors).ravel()
        lexical = 0.65 * w + 0.35 * c
        probs = {x["intent"]: x["probability"] for x in intent.top_predictions}
        pmax = max(probs.values(), default=0.0)
        labels = a2.kb.intent.astype(str).to_numpy()
        prior = np.array([probs.get(label, 0.0) / pmax if pmax else 0.0 for label in labels])
        scores = lexical * (0.85 + 0.15 * prior)
        order = np.argsort(scores)[::-1][:max(1, top_k)]
        evidence = []
        for i in order:
            row = a2.kb.iloc[int(i)]
            stable = hashlib.sha256((str(row.instruction) + "\0" + str(row.intent)).encode("utf-8")).hexdigest()[:16]
            evidence.append(Evidence(
                evidence_id=f"kb-{stable}", instruction=str(row.instruction), intent=str(row.intent),
                category=str(row.get("category", self.agent1.intent_to_category.get(str(row.intent), "UNKNOWN"))),
                response=str(row.response), word_similarity=float(w[i]), character_similarity=float(c[i]),
                retrieval_score=float(scores[i]),
            ))
        score = evidence[0].retrieval_score if evidence else 0.0
        # Explicit, retrieval-weighted disagreement diagnostic; it is not a probability.
        denom = sum(max(e.retrieval_score, 0.0) for e in evidence)
        predicted_intent_support = (sum(max(e.retrieval_score, 0.0) for e in evidence if e.intent == intent.intent) / denom) if denom else 0.0
        disagreement = (1.0 - predicted_intent_support) if self.debate_enabled else 0.0
        flags = [any(term in e.response.lower() for term in ("cannot", "unable", "not possible", "no")) for e in evidence]
        contradictions = len(set(flags)) > 1 and len(flags) > 1 and len({e.intent for e in evidence}) > 1
        sufficient = score >= a2.min_similarity
        return RetrievalResult(
            candidate_response=evidence[0].response if evidence else "", retrieval_score=score,
            sufficient=sufficient, evidence=evidence, disagreement_score=disagreement,
            disagreement=bool(self.debate_enabled and disagreement >= 0.50), contradictory_evidence=contradictions,
        )

    def process_query(self, query: Any, context: Any = None, query_id: str | None = None, top_k: int = 3, _intent_probabilities=None) -> dict[str, Any]:
        started = time.perf_counter()
        trace_id = str(uuid.uuid4())
        safe_query = query if isinstance(query, str) else ""
        trace: list[dict[str, Any]] = []
        def record(agent: str, received: dict[str, Any], decided: dict[str, Any], elapsed: float):
            trace.append({"agent": agent, "received": received, "decided": decided, "duration_ms": round(elapsed * 1000, 3), "external_api_called": False, "local_fallback_used": True})
        t = time.perf_counter(); intent = self._analyze(safe_query, _intent_probabilities); record("agent1", {"query": safe_query}, intent.__dict__, time.perf_counter()-t)
        t = time.perf_counter(); retrieval = self._retrieve(safe_query, intent, top_k); record("agent2", {"query": safe_query, "intent": intent.__dict__}, retrieval.__dict__, time.perf_counter()-t)
        t = time.perf_counter(); routing = self.agent3.plan(safe_query, intent, retrieval, context); record("agent3", {"intent": intent.__dict__, "retrieval": retrieval.__dict__}, routing.__dict__, time.perf_counter()-t)
        # One bounded retry broadens the evidence set once; there is no agent loop.
        if routing.recommendation == "RETRIEVE":
            t = time.perf_counter(); retry = self._retrieve(safe_query, intent, min(max(top_k + 2, 5), 10))
            record("agent2_retry", {"reason": routing.reason, "top_k": min(max(top_k + 2, 5), 10)}, retry.__dict__, time.perf_counter()-t)
            retrieval = retry
            t = time.perf_counter(); routing = self.agent3.plan(safe_query, intent, retrieval, context)
            record("agent3_retry", {"intent": intent.__dict__, "retrieval": retrieval.__dict__}, routing.__dict__, time.perf_counter()-t)
        t = time.perf_counter(); audit = self.agent4.audit(safe_query, intent, retrieval, routing); record("agent4", {"retrieval": retrieval.__dict__, "routing": routing.__dict__}, audit.__dict__, time.perf_counter()-t)
        t = time.perf_counter(); decision, response, explanation = self.agent5.decide(intent, retrieval, routing, audit)
        final = {"decision": decision, "response": response, "explanation": explanation}
        if decision == "ESCALATE":
            final["escalation_package"] = {
                "query": safe_query,
                "likely_intent": intent.intent,
                "category": intent.category,
                "missing_information": routing.missing_information,
                "supporting_evidence_ids": [e.evidence_id for e in retrieval.evidence],
                "reason": explanation,
            }
        record("agent5", {"routing": routing.__dict__, "audit": audit.__dict__}, final, time.perf_counter()-t)
        latency = (time.perf_counter()-started)*1000
        agents = {"agent1": intent.__dict__, "agent2": retrieval.__dict__, "agent3": routing.__dict__, "agent4": audit.__dict__, "agent5": final}
        return OrchestrationResult(query_id, safe_query, intent.intent, intent.category, intent.confidence, decision, response, agents, trace_id, round(latency, 3), trace).to_dict()

    def process_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch the classifier pass once, then run bounded evidence/routing per query."""
        if not items: return []
        queries = [item.get("query") if isinstance(item.get("query"), str) else "" for item in items]
        probabilities = self.agent1.model.predict_proba(queries)
        return [self.process_query(query, context=item.get("context"), query_id=item.get("query_id"), _intent_probabilities=probabilities[i])
                for i, (item, query) in enumerate(zip(items, queries))]
