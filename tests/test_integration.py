import unittest
from pathlib import Path

from orchestrator import FiveAgentSystem

MODEL = Path(__file__).resolve().parents[1] / "agent_models" / "two_agent_system.joblib"


@unittest.skipUnless(MODEL.is_file(), "Train the model bundle to run integration coverage")
class FiveAgentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = FiveAgentSystem.from_joblib(str(MODEL))

    def test_answer_clarify_escalate_paths_and_full_trace(self):
        answer = self.system.process_query("How do I track my package?", query_id="I1")
        clarify = self.system.process_query("Money was deducted but my order failed", query_id="I2")
        escalate = self.system.process_query("Can you help with my tax return?", query_id="I3")
        self.assertEqual([answer["decision"], clarify["decision"], escalate["decision"]], ["ANSWER", "CLARIFY", "ESCALATE"])
        self.assertEqual(set(answer["agents"]), {"agent1", "agent2", "agent3", "agent4", "agent5"})
        self.assertEqual(len(answer["trace"]), 5)
        self.assertIn("escalation_package", escalate["agents"]["agent5"])

    def test_top_three_mapping_and_typo_robustness(self):
        analysis = self.system._analyze("Where is my order?")
        self.assertEqual(len(analysis.top_predictions), 3)
        self.assertEqual(analysis.category, "ORDER")
        self.assertGreaterEqual(analysis.top_predictions[0]["probability"], analysis.top_predictions[1]["probability"])
        typo = self.system.process_query("wher is my ordar and traking numbr?")
        self.assertIn(typo["decision"], {"ANSWER", "CLARIFY", "ESCALATE"})
        self.assertEqual(len(typo["agents"]["agent2"]["evidence"]), 3)

    def test_batch_preserves_query_ids_and_order(self):
        rows = self.system.process_batch([{"query":"Cancel it","query_id":"Q1"},{"query":"How do I track my package?","query_id":"Q2"}])
        self.assertEqual([x["query_id"] for x in rows], ["Q1", "Q2"])


if __name__ == "__main__": unittest.main()
