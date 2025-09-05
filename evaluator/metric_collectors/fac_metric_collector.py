import os
import asyncio
import json
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.utils.module_extractor import register_metric_collector
from evaluator.utils.tool_logger import ToolLogger
from dotenv import load_dotenv

# Import our FAC evaluation components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fac_only_evaluation import FACOnlyEvaluation

load_dotenv()


@register_metric_collector("fac_metric_collector")
class FACMetricCollector(MetricCollector):
    """
    Final Answer Correctness (FAC) Metric Collector
    
    Evaluates LangGraph ReAct agents using the StableToolBench FAC evaluation framework.
    Uses OpenShift hosted models as the judge for evaluation.
    """
    
    def __init__(self, settings: Dict):
        super().__init__(settings)
        
        # FAC evaluation components
        self.fac_evaluator = None
        self.results = None
        
        # Metrics to collect
        self.total_queries = 0
        self.solved_queries = 0
        self.unsolved_queries = 0
        self.solve_rate = 0.0
        self.average_latency = 0.0
        self.total_latency = 0.0
        
        # Results storage
        self.raw_results = []
        self.converted_results = []
        self.fac_evaluation_results = None

    def get_collected_metrics_names(self) -> List[str]:
        return [
            "FAC Solve Rate (%)",
            "Total Queries",
            "Solved Queries", 
            "Unsolved Queries",
            "Average Latency (s)",
            "Total Latency (s)"
        ]

    def set_up(self) -> None:
        """Initialize the FAC evaluation system."""
        super().set_up()
        
        print("🚀 Setting up FAC Metric Collector...")
        
        # Initialize FAC evaluator
        self.fac_evaluator = FACOnlyEvaluation()
        
        # Reset metrics
        self.total_queries = 0
        self.solved_queries = 0
        self.unsolved_queries = 0
        self.solve_rate = 0.0
        self.average_latency = 0.0
        self.total_latency = 0.0
        self.raw_results = []
        self.converted_results = []
        self.fac_evaluation_results = None
        
        print("✅ FAC Metric Collector setup complete")

    def prepare_for_measurement(self, query: str) -> None:
        """Prepare for measuring a single query (not used in batch mode)."""
        # For FAC evaluation, we typically run in batch mode
        # This method is called for each query but we collect results in register_measurement
        pass

    def register_measurement(self, query: str, **kwargs) -> None:
        """Register measurement for a single query."""
        # For FAC evaluation, we typically run the full evaluation in set_up
        # and collect results in report_results. This method is for compatibility.
        pass

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run the complete FAC evaluation on all synthetic queries."""
        if not self.fac_evaluator:
            raise RuntimeError("FAC evaluator not initialized. Call set_up() first.")
        
        print("🔍 Running full FAC evaluation...")
        
        # Run the complete evaluation
        results = await self.fac_evaluator.run_complete_evaluation()
        
        # Extract metrics from results
        self.total_queries = results.get('total_queries', 0)
        self.solved_queries = results.get('solved_queries', 0)
        self.unsolved_queries = results.get('unsolved_queries', 0)
        self.solve_rate = results.get('solve_rate', 0.0)
        self.average_latency = results.get('average_latency', 0.0)
        self.total_latency = results.get('total_latency', 0.0)
        
        # Store detailed results
        self.raw_results = results.get('raw_results', [])
        self.converted_results = results.get('converted_results', [])
        self.fac_evaluation_results = results.get('fac_evaluation_results', None)
        
        # Mark collection as complete
        self.collection_active = False
        
        print(f"✅ FAC evaluation complete: {self.solve_rate:.1f}% solve rate")
        
        return results

    def tear_down(self) -> None:
        """Clean up resources."""
        super().tear_down()
        print("🧹 FAC Metric Collector torn down")

    def report_results(self) -> Dict[str, Any] or None:
        """Report the collected FAC metrics."""
        super().report_results()
        
        if self.total_queries == 0:
            raise RuntimeError("No FAC evaluation results available. Run evaluation first.")
        
        results = {
            "FAC Solve Rate (%)": self.solve_rate,
            "Total Queries": self.total_queries,
            "Solved Queries": self.solved_queries,
            "Unsolved Queries": self.unsolved_queries,
            "Average Latency (s)": self.average_latency,
            "Total Latency (s)": self.total_latency
        }
        
        # Add detailed results if available
        if self.fac_evaluation_results:
            results["Detailed Results"] = self.fac_evaluation_results
        
        return results

    def get_detailed_results(self) -> Dict[str, Any]:
        """Get detailed evaluation results including raw and converted answers."""
        return {
            "raw_results": self.raw_results,
            "converted_results": self.converted_results,
            "fac_evaluation_results": self.fac_evaluation_results,
            "metrics": self.report_results()
        }
