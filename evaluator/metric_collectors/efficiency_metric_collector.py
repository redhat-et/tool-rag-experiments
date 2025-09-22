import time
from typing import Any, Dict, List

from evaluator.components.data_provider import QuerySpecification
from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.utils.module_extractor import register_metric_collector


@register_metric_collector("efficiency_metric_collector")
class EfficiencyMetricCollector(MetricCollector):
    """
    Collects metric related to the efficiency of inference.
    In the future, this metric collector may implement other related metrics such as:
    - separate planner latency, tool I/O latency, etc.
    - latency percentiles
    - seconds per success
    - tokens per success
    - cost metrics
    """

    def __init__(self, settings: Dict):
        super().__init__(settings)

        self._rows = []
        self._n_queries = None
        self._total_latency = None

        self.start_time = None

    def get_collected_metrics_names(self) -> List[str]:
        return ["Average Latency (s)"]

    def set_up(self) -> None:
        super().set_up()

        self._rows.clear()
        self._n_queries = 0
        self._total_latency = 0

    def prepare_for_measurement(self, query_spec: QuerySpecification) -> None:
        self.start_time = time.time()

    def register_measurement(self, query_spec: QuerySpecification, **kwargs) -> None:
        self._n_queries += 1

        end_time = time.time()
        response_time = end_time - self.start_time
        self._total_latency += response_time

    def tear_down(self) -> None:
        super().tear_down()

    def report_results(self) -> Dict[str, Any] or None:
        super().report_results()

        if self._n_queries == 0:
            raise RuntimeError("No measurements registered, cannot produce results.")

        average_latency = self._total_latency / self._n_queries
        print(f"Average Latency (s): {average_latency}")
        return {"Average Latency (s)": average_latency}
