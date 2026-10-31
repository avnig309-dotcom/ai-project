import unittest
from types import SimpleNamespace

import api


class StubSystem:
    def _analyze(self, query):
        return SimpleNamespace(query=query, intent="track_order", category="ORDER", confidence=.9,
                               margin=.5, uncertain=False, top_predictions=[], out_of_domain=False,
                               ood_method="test", model_version="test")

    def process_query(self, query, context=None, query_id=None):
        return {"query_id": query_id, "query": query, "predicted_intent": "track_order",
                "predicted_category": "ORDER", "confidence": .9, "decision": "ANSWER",
                "response": "Use order tracking.", "agents": {}, "trace_id": "test", "latency_ms": 1,
                "trace": []}

    def process_batch(self, items):
        return [self.process_query(x["query"], x.get("context"), x.get("query_id")) for x in items]


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.original = api.system
        api.system = StubSystem()

    def tearDown(self):
        api.system = self.original

    def test_openapi_and_health_routes_exist(self):
        paths = api.app.openapi()["paths"]
        self.assertTrue({"/health", "/ready", "/api/v1/config", "/api/v1/analyze", "/api/v1/query", "/api/v1/batch"}.issubset(paths))
        self.assertEqual(api.health()["status"], "ok")

    def test_query_endpoint_contract(self):
        result = api.query(api.QueryRequest(query="track my order", query_id="Q1"))
        self.assertEqual(result["decision"], "ANSWER")
        self.assertEqual(result["query_id"], "Q1")

    def test_batch_endpoint_contract(self):
        result = api.batch(api.BatchRequest(queries=[api.QueryRequest(query="q1"), api.QueryRequest(query="q2")]))
        self.assertEqual(len(result["results"]), 2)


if __name__ == "__main__": unittest.main()
