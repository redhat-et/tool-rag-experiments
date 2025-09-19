from typing import Any, Dict, List, Optional, Set
import math

from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.utils.module_extractor import register_metric_collector


@register_metric_collector("tool_retrieval_metric_collector")
class ToolRetrievalMetricCollector(MetricCollector):
    """
    Collects and calculates metrics related to the quality of tool retrieval (before any tool calling).

    Expected kwargs to register_measurement:
      - executed_tools: List[str]
      - retrieved_tools: Optional[List[str]]  (ranked list; None allowed for baseline)

    Config:
      - ks: list of cutoffs for @K metrics (default [1,3,5,10])
      - ap_rel_threshold: relevance >= threshold counts as "relevant" for AP/MRR/hit/recall (default 1.0)
      - normalize_tool_id: normalization/canonicalization for tool IDs

    Notes:
      - If retrieved_tools is None, it is treated as an empty ranked list (baseline).
      - nDCG@K uses binary relevance.
      - MAP/AP uses binary relevance with threshold (ap_rel_threshold).
    """

    def __init__(self, settings: Dict):
        super().__init__(settings)

        self.ks = self._settings["ks"]
        self.ap_rel_threshold = self._settings["ap_rel_threshold"]

        self._rows = []
        self._num_queries_with_retrieval = None

    def get_collected_metrics_names(self) -> List[str]:
        out = []

        # Per-K aggregates
        for K in self.ks:
            out.extend([f"hit_at{K}_rate", f"recall_at{K}_mean", f"precision_at{K}_mean", f"ndcg_at{K}_mean"])

        # Global aggregates
        out.extend(["mrr_mean", "map_mean"])

        return out

    def set_up(self) -> None:
        self._rows.clear()
        self._num_queries_with_retrieval = 0

    def prepare_for_measurement(self, query: str) -> None:
        pass

    def register_measurement(self, query: str, **kwargs) -> None:
        if "executed_tools" not in kwargs:
            raise ValueError(f"{self.get_name()}: Mandatory parameter 'executed_tools' was not provided.")
        executed_tools = kwargs["executed_tools"]

        if "retrieved_tools" not in kwargs:
            raise ValueError(f"{self.get_name()}: Mandatory parameter 'retrieved_tools' was not provided.")
        retrieved = kwargs["retrieved_tools"]

        if retrieved is None:
            # can happen if we are running the RAG-less baseline
            return

        self._num_queries_with_retrieval += 1

        gold_set = set(executed_tools)

        # Build graded relevance list for retrieved items (post)
        rels_post = [self._grade(r, gold_set) for r in retrieved]

        # @K metrics
        per_k = {}
        for K in self.ks:
            hit = 1.0 if any(self._is_relevant(x) for x in rels_post[:K]) else 0.0
            recall = self._safe_div(self._count_relevant(rels_post[:K]), len(gold_set))
            precision = self._safe_div(self._count_relevant(rels_post[:K]), min(K, len(retrieved)))
            ndcg = self._ndcg_at_k(rels_post, K, ideal_rel_count=len(gold_set))
            per_k[K] = {"hit": hit, "recall": recall, "precision": precision, "ndcg": ndcg}

        # MRR & AP/MAP (binary by threshold)
        mrr = self._mrr(rels_post)
        ap = self._ap(rels_post, len(gold_set))

        row = {
            "per_k": per_k,
            "mrr": mrr,
            "ap": ap,
        }
        self._rows.append(row)

    def tear_down(self) -> None:
        pass

    def report_results(self) -> Dict[str, Any]:
        if self._num_queries_with_retrieval == 0:
            # the current run is on the baseline without RAG - retrieval metrics are not available
            keys = self.get_collected_metrics_names()
            return {key: "N/A" for key in keys}

        out = {}

        # Per-K aggregates
        for K in self.ks:
            out[f"hit_at{K}_rate"] = self._avg_k(lambda r: r["per_k"][K]["hit"])
            out[f"recall_at{K}_mean"] = self._avg_k(lambda r: r["per_k"][K]["recall"])
            out[f"precision_at{K}_mean"] = self._avg_k(lambda r: r["per_k"][K]["precision"])
            out[f"ndcg_at{K}_mean"] = self._avg_k(lambda r: r["per_k"][K]["ndcg"])

        # Global aggregates
        out["mrr_mean"] = self._avg(lambda r: r["mrr"])
        out["map_mean"] = self._avg(lambda r: r["ap"])

        return out

    # ---- helpers ----

    def _avg_k(self, get):
        vals = []
        for r in self._rows:
            v = get(r)
            if isinstance(v, (int, float)):
                vals.append(v)
        return round(sum(vals) / len(vals), 6) if vals else None

    def _avg(self, get):
        vals = [get(r) for r in self._rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 6) if vals else None

    @staticmethod
    def _grade(candidate: str, gold_set: Set[str]) -> float:
        if not gold_set:
            return 0.0
        return 1.0 if candidate in gold_set else 0.0

    def _is_relevant(self, rel: float) -> bool:
        return rel >= self.ap_rel_threshold

    def _count_relevant(self, rels: List[float]) -> int:
        return sum(1 for r in rels if self._is_relevant(r))

    @staticmethod
    def _safe_div(num: int | float, den: int | float) -> float:
        return float(num) / float(den) if den else 0.0

    @staticmethod
    def _dcg_at_k(rels: List[float], k: int) -> float:
        s = 0.0
        for i, r in enumerate(rels[:k], start=1):
            # standard nDCG formula with graded gains
            s += (2.0**float(r) - 1.0) / math.log2(i + 1.0)
        return s

    def _ndcg_at_k(self, rels: List[float], k: int, ideal_rel_count: int) -> Optional[float]:
        if ideal_rel_count <= 0:
            return None
        dcg = self._dcg_at_k(rels, k)
        # Ideal DCG assumes perfect ranking of up to min(K, ideal_rel_count) items with rel=1.0
        ideal_rels = [1.0] * min(k, ideal_rel_count)
        idcg = self._dcg_at_k(ideal_rels, k)
        if idcg <= 0:
            return None
        return dcg / idcg

    def _mrr(self, rels: List[float]) -> Optional[float]:
        # reciprocal rank of first item with rel >= threshold
        for idx, r in enumerate(rels, start=1):
            if self._is_relevant(r):
                return 1.0 / idx
        return 0.0  # if gold exists but never retrieved, MRR is 0

    def _ap(self, rels: List[float], n_gold: int) -> Optional[float]:
        if n_gold <= 0:
            return None
        num_rel_seen = 0
        sum_precision = 0.0
        for i, r in enumerate(rels, start=1):
            if self._is_relevant(r):
                num_rel_seen += 1
                sum_precision += num_rel_seen / i
        # divide by actual number of gold items (AP definition)
        return sum_precision / n_gold
