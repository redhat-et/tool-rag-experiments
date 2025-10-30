from typing import List, Dict, Any

from evaluator.components.data_provider import QuerySpecification
from evaluator.config.schema import ModelConfig
from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.utils.module_extractor import register_metric_collector
from evaluator.utils.utils import log_verbose, log


@register_metric_collector("tool_selection_metric_collector")
class ToolSelectionMetricCollector(MetricCollector):
    """
    Tool Selection Metric Collector validates executed tool sequences against golden sets from the dataset.
    The comparison is order-independent - the metric collector only cares which tools were called, not the order.
    """
    def __init__(self, settings: Dict, model_config: List[ModelConfig]):
        super().__init__(settings, model_config)

        self.total_queries = 0
        self.exact_matches = 0
        self.precision_sum = 0.0
        self.recall_sum = 0.0

    def get_collected_metrics_names(self) -> List[str]:
        return ["Exact Tool Selection Match Rate",
                "Tool Selection Precision",
                "Tool Selection Recall",
                "Spurious Tool Calling Rate"]

    def set_up(self) -> None:
        super().set_up()

        self.total_queries = 0
        self.exact_matches = 0
        self.precision_sum = 0.0
        self.recall_sum = 0.0

    def prepare_for_measurement(self, query_spec: QuerySpecification) -> None:
        pass

    @staticmethod
    def _compute_tool_set_metrics(golden_tools: List[str], executed_tools: List[str]) -> Dict[str, Any]:
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

        return {
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall
        }

    def register_measurement(self, query_spec: QuerySpecification, **kwargs) -> None:
        if "executed_tools" not in kwargs:
            raise ValueError(f"{self.get_name()}: Mandatory parameter 'executed_tools' was not provided.")
        executed_tool_names = kwargs["executed_tools"]

        if not query_spec.golden_tools:
            log_verbose(f"{self.get_name()}: No golden tools specified, skipping this query.")
            return
        golden_tool_names = list(query_spec.golden_tools.keys())

        log_verbose(f"Golden tools for the query: {golden_tool_names}\nActually executed tools: {executed_tool_names}\n")
        metrics = self._compute_tool_set_metrics(golden_tool_names, executed_tool_names)

        self.total_queries += 1
        if metrics["exact_match"]:
            self.exact_matches += 1
        self.precision_sum += metrics["precision"]
        self.recall_sum += metrics["recall"]

    def tear_down(self) -> None:
        super().tear_down()

    def report_results(self) -> Dict[str, Any] or None:
        super().report_results()

        if self.total_queries == 0:
            raise RuntimeError("No measurements registered, cannot produce results.")

        results = {
            "Exact Tool Selection Match Rate": (
                (self.exact_matches or 0) / (self.total_queries or 1)
                if self.total_queries else 0.0
            ),
            "Tool Selection Precision": self.precision_sum / self.total_queries,
            "Tool Selection Recall": self.recall_sum / self.total_queries,
            "Spurious Tool Calling Rate": 1.0 - (self.precision_sum / self.total_queries),
        }

        for key, value in results.items():
            log(f"{key}: {value:.3f}")
        return results
