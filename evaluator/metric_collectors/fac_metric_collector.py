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


@register_metric_collector("fac_metric_collector")
class FACMetricCollector(MetricCollector):
    """
    Final Answer Correctness (FAC) Metric Collector - Framework Integrated Version
    
    Collects FAC metrics during the framework's query-by-query experiment flow.
    This version integrates with the experiment framework rather than running standalone.
    """
    
    def __init__(self, settings: Dict):
        super().__init__(settings)
        self.logger = ToolLogger("fac_metric_collector.log")
        
        # Metrics storage
        self.query_results = []
        self.total_queries = 0
        self.solved_queries = 0
        self.total_latency = 0.0
        
        # StableToolBench integration
        self.stabletoolbench_results = None

    def get_collected_metrics_names(self) -> List[str]:
        return [
            "FAC Solve Rate (%)",
            "FAC Total Queries",
            "FAC Solved Queries", 
            "FAC Unsolved Queries",
            "FAC Average Latency (s)",
            "FAC Total Latency (s)"
        ]

    def set_up(self) -> None:
        """Initialize the FAC metric collector."""
        super().set_up()
        print("🚀 Setting up FAC Metric Collector...")
        print("✅ FAC Metric Collector setup complete")

    def prepare_for_measurement(self, query: str) -> None:
        """Prepare for measuring a single query."""
        # No preparation needed - we'll analyze the response from the algorithm
        pass

    def register_measurement(self, query: str, response: Any = None, correct_tool: str = None) -> None:
        """Register measurement for a single query."""
        import time
        
        start_time = time.time()
        
        try:
            # Extract final answer from algorithm response
            final_answer = self.extract_final_answer_from_response(response)
            latency = time.time() - start_time
            
            result = {
                "query": query,
                "response": response,
                "final_answer": final_answer,
                "latency": latency,
                "correct_tool": correct_tool
            }
            
            self.total_latency += latency
            self.query_results.append(result)
            
            print(f"📊 FAC Query Collected: {query[:50]}... (latency: {latency:.2f}s)")
            
        except Exception as e:
            latency = time.time() - start_time
            result = {
                "query": query,
                "response": response,
                "final_answer": "",
                "latency": latency,
                "error": str(e)
            }
            self.total_latency += latency
            self.query_results.append(result)
            print(f"❌ FAC Query Error: {e}")

    def extract_final_answer_from_response(self, response: Any) -> str:
        """Extract final answer from algorithm response."""
        try:
            # Handle different response types from algorithms
            if response is None:
                return ""
            
            # If it's a string, return it directly
            if isinstance(response, str):
                return response.strip()
            
            # If it's a dict, look for common answer fields
            if isinstance(response, dict):
                # Try common answer field names
                for field in ['answer', 'output', 'result', 'response', 'content']:
                    if field in response:
                        return str(response[field]).strip()
                
                # If no common field, return the whole dict as string
                return str(response)
            
            # For other types, convert to string
            return str(response).strip()
                
        except Exception as e:
            print(f"❌ Error extracting final answer: {e}")
            return str(response) if response else ""


    def tear_down(self) -> None:
        """Clean up after all measurements."""
        print("🧹 Starting FAC Metric Collector tear down...")
        
        # Run StableToolBench evaluation if we have results
        if self.query_results:
            self.run_stabletoolbench_evaluation()
        
        # Mark collection as complete
        self.collection_active = False
        print(f"🧹 Collection marked as complete: {self.collection_active}")
        
        super().tear_down()
        print("🧹 FAC Metric Collector torn down")

    def run_stabletoolbench_evaluation(self):
        """Run StableToolBench evaluation on collected results."""
        try:
            print("🔍 Running StableToolBench FAC evaluation...")
            
            # Save results in StableToolBench format
            self.save_results_for_stabletoolbench()
            
            # Run StableToolBench evaluation
            from .stabletoolbench_fac_eval import run_stabletoolbench_fac_eval
            success = run_stabletoolbench_fac_eval()
            
            if success:
                print("✅ StableToolBench FAC evaluation completed!")
                self.stabletoolbench_results = {"success": True}
            else:
                print("❌ StableToolBench FAC evaluation failed!")
                self.stabletoolbench_results = {"success": False}
                
        except Exception as e:
            print(f"❌ Error running StableToolBench evaluation: {e}")
            self.stabletoolbench_results = {"success": False, "error": str(e)}

    def save_results_for_stabletoolbench(self):
        """Save results in StableToolBench format."""
        # Create output directories
        output_dir = Path("fac_evaluation_results")
        converted_dir = output_dir / "converted_answers" / "langgraph_react"
        converted_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to StableToolBench format
        converted_results = {}
        for i, result in enumerate(self.query_results):
            # Save all results for StableToolBench evaluation (it will determine if they're solved)
            query_id = f"synthetic_{i+1:03d}"
            converted_results[query_id] = {
                "query": result["query"],
                "answer": {
                    "final_answer": json.dumps({"final_answer": result["final_answer"]})
                }
            }
        
        # Save as G1_instruction.json
        converted_file = converted_dir / "G1_instruction.json"
        with open(converted_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"✅ Saved {len(converted_results)} results for StableToolBench evaluation")

    def report_results(self) -> Dict[str, Any]:
        """Report the collected metrics."""
        print(f"🔍 report_results called, collection_active: {self.collection_active}")
        if self.collection_active:
            raise RuntimeError(f"Metric collector {self.get_name()}: cannot report results while collection is active")
        
        if self.total_queries == 0:
            self.total_queries = len(self.query_results)
        
        if self.total_queries == 0:
            raise RuntimeError("No FAC evaluation results available. Run evaluation first.")
        
        # Use StableToolBench evaluation results
        if self.stabletoolbench_results and self.stabletoolbench_results.get("success"):
            # Parse the StableToolBench CSV results
            stabletoolbench_solve_rate = self.parse_stabletoolbench_results()
            if stabletoolbench_solve_rate is not None:
                solve_rate = stabletoolbench_solve_rate
                solved_queries = int((solve_rate / 100) * self.total_queries)
            else:
                # If StableToolBench evaluation failed, provide fallback results
                print("⚠️ StableToolBench evaluation failed. Using fallback results.")
                solve_rate = 0.0  # Conservative fallback
                solved_queries = 0
        else:
            # If StableToolBench evaluation wasn't run, provide fallback results
            print("⚠️ StableToolBench evaluation not completed. Using fallback results.")
            solve_rate = 0.0  # Conservative fallback
            solved_queries = 0
        
        average_latency = self.total_latency / self.total_queries if self.total_queries > 0 else 0
        
        results = {
            "FAC Solve Rate (%)": solve_rate,
            "FAC Total Queries": self.total_queries,
            "FAC Solved Queries": solved_queries,
            "FAC Unsolved Queries": self.total_queries - solved_queries,
            "FAC Average Latency (s)": average_latency,
            "FAC Total Latency (s)": self.total_latency
        }
        
        print(f"📊 FAC Results: {solve_rate:.1f}% solve rate, {average_latency:.2f}s avg latency")
        
        return results

    def parse_stabletoolbench_results(self) -> float:
        """Parse StableToolBench CSV results to get the actual solve rate."""
        try:
            import pandas as pd
            
            results_file = Path("fac_evaluation_results/evaluation/fac_results.csv")
            if not results_file.exists():
                print("⚠️ StableToolBench results file not found")
                return None
            
            df = pd.read_csv(results_file)
            
            # Count solved vs unsolved based on evaluation text
            total_queries = len(df)
            solved_count = 0
            
            for eval_text in df['evaluation']:
                if 'solved' in str(eval_text).lower():
                    solved_count += 1
            
            solve_rate = (solved_count / total_queries) * 100 if total_queries > 0 else 0
            
            print(f"📊 StableToolBench Results: {solved_count}/{total_queries} solved ({solve_rate:.1f}%)")
            return solve_rate
            
        except Exception as e:
            print(f"❌ Error parsing StableToolBench results: {e}")
            return None