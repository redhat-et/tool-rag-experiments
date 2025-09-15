import os
import time
from typing import Dict, List, Any

from evaluator.components.data_provider import QuerySpecification
from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.utils.module_extractor import register_metric_collector
from evaluator.utils.tool_logger import ToolLogger
from dotenv import load_dotenv

load_dotenv()


@register_metric_collector("basic_metric_collector")
class BasicMetricCollector(MetricCollector):
    def __init__(self, settings: Dict):
        super().__init__(settings)

        # Metric-related fields
        self.tool_execution_count = None
        self.correct_tool_count = None
        self.irrelevant_tool_count = None
        self.total_latency = None
        self.num_queries = None

        # Auxiliary fields
        self.start_time = None
        self.tool_logger = None

    def get_collected_metrics_names(self) -> List[str]:
        return ["Tool Execution Rate",
                "Correct Tool Rate",
                "Irrelevant Tool Rate",
                "Average Latency (s)"]

    def set_up(self) -> None:
        super().set_up()

        self.tool_execution_count = 0
        self.correct_tool_count = 0
        self.irrelevant_tool_count = 0
        self.total_latency = 0
        self.num_queries = 0

        self.tool_logger = ToolLogger(os.getenv("TOOL_LOG_PATH"))

    def prepare_for_measurement(self, query_spec: QuerySpecification) -> None:
        self.start_time = time.time()

    def register_measurement(self, query_spec: QuerySpecification, **kwargs) -> None:
        if not query_spec.golden_tools:
            print(f"{self.get_name()}: No golden tools specified, skipping this query.")
            return

        # if there are multiple tools in this set, only one will be considered here
        correct_tool = list(query_spec.golden_tools.keys())[0]

        end_time = time.time()
        response_time = end_time - self.start_time
        self.total_latency += response_time

        executed_tools = self.tool_logger.get_executed_tools()
        num_correct_tool_used = correct_tool in executed_tools
        self.tool_execution_count += 1 if executed_tools and executed_tools[0] != "unknown" else 0
        self.correct_tool_count += 1 if num_correct_tool_used else 0
        self.irrelevant_tool_count += len(executed_tools) - num_correct_tool_used

        self.num_queries += 1

    def tear_down(self) -> None:
        super().tear_down()

    def report_results(self) -> Dict[str, Any] or None:
        super().report_results()

        if self.num_queries == 0:
            raise RuntimeError("No measurements registered, cannot produce results.")

        results = {"Tool Execution Rate": self.tool_execution_count / self.num_queries,
                   "Correct Tool Rate": self.correct_tool_count / self.num_queries,
                   "Irrelevant Tool Rate": self.irrelevant_tool_count / self.num_queries,
                   "Average Latency (s)": self.total_latency / self.num_queries}

        return results
