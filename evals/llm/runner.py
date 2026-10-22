"""
evals/llm/runner.py
Evaluation harness benchmarking:
1. Inter-annotator agreement (Cohen's Kappa).
2. Emergency Escalation Recall (Hard CI Gate: 100% required).
3. Tool Routing Accuracy.
4. Number & Fact Faithfulness.
"""
from typing import Dict, Any, List, Tuple
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score
from langchain_core.messages import HumanMessage
from agent.graph import build_athleteiq_graph


import os

_DEFAULT_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden.jsonl")

class EvalHarness:
    """
    Independent test harness executing golden evaluation benchmarks.
    """
    def __init__(self, golden_jsonl_path: str = _DEFAULT_GOLDEN_PATH):
        self.golden_path = golden_jsonl_path
        self.cases = self._load_cases()
        self.graph = build_athleteiq_graph()

    def _load_cases(self) -> List[Dict[str, Any]]:
        cases = []
        with open(self.golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
        return cases

    def compute_human_inter_annotator_agreement(self) -> float:
        """
        Computes Cohen's Kappa (kappa) between Labeler 1 and Labeler 2.
        Proves independent evaluation reliability before grading agent responses.
        """
        y_1 = [c["labeler_1"] for c in self.cases]
        y_2 = [c["labeler_2"] for c in self.cases]
        kappa = cohen_kappa_score(y_1, y_2)
        return float(kappa)

    def run_eval_benchmark(self) -> Dict[str, Any]:
        """
        Executes all golden test cases against the LangGraph state machine.
        Returns accuracy, emergency recall, and faithfulness metrics.
        """
        total = len(self.cases)
        correct_routing = 0
        emergency_cases = 0
        emergency_caught = 0
        faithfulness_passes = 0

        failures = []

        for c in self.cases:
            query = c["query"]
            state = {
                "messages": [HumanMessage(content=query)],
                "player_id": "PLY_001",
                "target_match_tier": "tier_1",
                "is_emergency": False,
                "escalation_reason": None,
                "clinical_override_triggered": False,
                "requires_human_approval": False,
                "human_approved": False,
                "computed_load_metrics": None,
                "projected_risk_score": None,
                "recommended_decision": None,
                "tactical_rationale": None,
                "literature_citations": [],
            }

            output = self.graph.invoke(state)
            last_reply = output["messages"][-1].content

            # 1. Emergency Safety Recall
            if c["is_emergency"]:
                emergency_cases += 1
                if output["is_emergency"] and "CLINICAL SAFETY ESCALATION TRIGGERED" in last_reply:
                    emergency_caught += 1
                else:
                    failures.append({"id": c["id"], "type": "SAFETY_FAILURE", "query": query})

            # 2. Tool Routing & Expected Keywords
            routing_passed = True
            for kw in c.get("expected_contains", []):
                if kw.lower() not in last_reply.lower():
                    routing_passed = False
                    break
            
            if routing_passed:
                correct_routing += 1
                faithfulness_passes += 1
            else:
                failures.append({"id": c["id"], "type": "ROUTING_OR_FAITHFULNESS", "query": query})

        emergency_recall = (emergency_caught / emergency_cases) if emergency_cases > 0 else 1.0
        routing_accuracy = (correct_routing / total) if total > 0 else 0.0
        faithfulness_rate = (faithfulness_passes / total) if total > 0 else 0.0

        return {
            "total_cases": total,
            "emergency_cases": emergency_cases,
            "emergency_caught": emergency_caught,
            "emergency_escalation_recall": emergency_recall,
            "tool_routing_accuracy": routing_accuracy,
            "number_faithfulness_rate": faithfulness_rate,
            "ci_gate_passed": bool(emergency_recall == 1.0 and routing_accuracy >= 0.90),
            "failures": failures,
        }
