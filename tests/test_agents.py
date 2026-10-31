import unittest

from schemas import AuditResult, IntentResult, RetrievalResult, RoutingResult
from support_agents import Coordinator, ResponseAuditor, RoutingAgent, detect_out_of_domain
from providers import MockProvider, OrganizerAPIProvider


class DecisionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.intent = IntentResult("How do I track an order?", "track_order", "ORDER", confidence=.9, margin=.7, uncertain=False)
        self.retrieval = RetrievalResult("Use the order page to track it.", .8, True)
        self.routing = RoutingResult("PROCEED", "Evidence is sufficient.", risk_level="MEDIUM", sufficient_information=True)

    def test_approved_grounded_answer(self):
        audit = AuditResult("APPROVE", safe_response=self.retrieval.candidate_response)
        self.assertEqual(Coordinator().decide(self.intent, self.retrieval, self.routing, audit)[0], "ANSWER")

    def test_missing_information_clarifies(self):
        routing = RoutingResult("CLARIFY", "Need reference", ["order reference"])
        audit = AuditResult("CLARIFY")
        decision, response, _ = Coordinator().decide(self.intent, self.retrieval, routing, audit)
        self.assertEqual(decision, "CLARIFY")
        self.assertIn("order reference", response)

    def test_critical_audit_escalates(self):
        audit = AuditResult("ESCALATE", issues=[{"code": "UNVERIFIED_TRANSACTION_CLAIM", "severity": "critical"}])
        self.assertEqual(Coordinator().decide(self.intent, self.retrieval, self.routing, audit)[0], "ESCALATE")

    def test_auditor_rejects_unverified_refund_claim(self):
        risky = RetrievalResult("Your refund has been processed.", .9, True)
        result = ResponseAuditor().audit("Where is my refund?", self.intent, risky, self.routing)
        self.assertEqual(result.status, "ESCALATE")
        self.assertTrue(any(x["code"] == "UNVERIFIED_TRANSACTION_CLAIM" for x in result.issues))

    def test_multiple_requests_clarify(self):
        result = RoutingAgent().plan("Cancel my order and track my refund", self.intent, self.retrieval)
        self.assertEqual(result.recommendation, "CLARIFY")

    def test_empty_query_clarifies_deterministically(self):
        result = RoutingAgent().plan("", IntentResult("", None, None), RetrievalResult("", 0.0, False))
        self.assertEqual(result.recommendation, "CLARIFY")

    def test_ambiguous_cancellation_requests_reference(self):
        intent = IntentResult("Cancel it", "cancel_order", "CANCEL", confidence=.8, margin=.4, uncertain=False)
        result = RoutingAgent().plan("Cancel it", intent, self.retrieval)
        self.assertEqual(result.recommendation, "CLARIFY")
        self.assertIn("order reference", result.missing_information)

    def test_deducted_payment_and_failed_order_requests_details(self):
        query = "Money was deducted but my order failed"
        intent = IntentResult(query, "payment_issue", "PAYMENT", confidence=.9, margin=.6, uncertain=False)
        result = RoutingAgent().plan(query, intent, self.retrieval)
        self.assertEqual(result.recommendation, "CLARIFY")
        self.assertIn("payment reference", result.missing_information)

    def test_uncertain_or_out_of_scope_escalates_or_clarifies(self):
        outside = IntentResult("unrelated", "other", "OTHER", out_of_domain=True)
        result = RoutingAgent().plan("unrelated", outside, self.retrieval)
        self.assertEqual(result.recommendation, "ESCALATE")

    def test_known_unrelated_domain_cues_are_out_of_scope(self):
        self.assertTrue(detect_out_of_domain("Can you help with my tax return?"))
        self.assertFalse(detect_out_of_domain("How can I track my order?"))

    def test_disagreement_is_escalated(self):
        retrieval = RetrievalResult("other guidance", .7, True, disagreement=True, disagreement_score=.8)
        result = RoutingAgent().plan(self.intent.query, self.intent, retrieval)
        self.assertEqual(result.recommendation, "ESCALATE")

    def test_provider_modes_are_explicit(self):
        mock = MockProvider().audit()
        self.assertEqual(mock["mode"], "mock")
        self.assertTrue(mock["response"]["mock"])
        with self.assertRaisesRegex(RuntimeError, "official API documentation"):
            OrganizerAPIProvider().audit()


if __name__ == "__main__": unittest.main()
