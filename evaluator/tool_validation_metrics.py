"""
Tool Sequence Validation Metrics

Validates executed tool sequences against golden sequences from StableToolBench data.
Order-independent comparison - only cares which tools were called, not the order.
"""
import csv
import os
from typing import List, Dict, Any


def compute_tool_set_metrics(golden_tools: List[str], executed_tools: List[str]) -> Dict[str, Any]:
    """
    Compute order-independent tool set metrics.
    Order doesn't matter - only which tools were called.
    
    Args:
        golden_tools: Expected tools from StableToolBench "relevant APIs"
        executed_tools: Actually executed tools from tool logs
    
    Returns:
        Dict with metrics: exact_match, precision, recall, f1
    """
    golden_set = set(golden_tools or [])
    executed_set = set(executed_tools or [])
    
    # Tools that are both golden AND executed (correct selections)
    correct_tools = golden_set.intersection(executed_set)
    
    # Exact match: tool sets are identical (order independent)
    exact_match = (golden_set == executed_set)
    
    # Precision: fraction of executed tools that were correct
    if len(executed_set) > 0:
        precision = len(correct_tools) / len(executed_set)
    else:
        precision = 0.0
    
    # Recall: fraction of golden tools that were executed
    if len(golden_set) > 0:
        recall = len(correct_tools) / len(golden_set)
    else:
        recall = 1.0  # Perfect recall if no tools were needed
    
    # F1 score: harmonic mean of precision and recall
    if (precision + recall) > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1_score
    }


class ToolSetMetricsAggregator:
    """Aggregates order-independent tool set metrics across multiple queries."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_queries = 0
        self.exact_matches = 0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.f1_sum = 0.0
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add metrics from a single query to the aggregation."""
        self.total_queries += 1
        if metrics["exact_match"]:
            self.exact_matches += 1
        self.precision_sum += metrics["precision"]
        self.recall_sum += metrics["recall"]
        self.f1_sum += metrics["f1"]
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all queries."""
        if self.total_queries == 0:
            return {
                "exact_match_rate": 0.0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0
            }
        
        return {
            "exact_match_rate": self.exact_matches / self.total_queries,
            "avg_precision": self.precision_sum / self.total_queries,
            "avg_recall": self.recall_sum / self.total_queries,
            "avg_f1": self.f1_sum / self.total_queries
        }
    
    def save_to_csv(self, output_path: str = None):
        """Save aggregated metrics to CSV file."""
        if output_path is None:
            output_path = os.getenv("TOOL_SET_RESULTS_PATH", "tool_set_results.csv")
        
        aggregated = self.get_aggregated_metrics()
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Exact Match Rate",
                "Avg Precision", 
                "Avg Recall",
                "Avg F1"
            ])
            writer.writerow([
                f"{aggregated['exact_match_rate']:.3f}",
                f"{aggregated['avg_precision']:.3f}",
                f"{aggregated['avg_recall']:.3f}",
                f"{aggregated['avg_f1']:.3f}"
            ])


