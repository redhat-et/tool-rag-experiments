from typing import List

from evaluator.utils.module_extractor import Spec

EVALUATED_ALGORITHMS: List[Spec] = [
    ("no_tool_rag_baseline", {}),
    #("basic_tool_rag", {"top_k": 3}),
]

METRIC_COLLECTORS: List[Spec] = [
    ("basic_metric_collector", {}),
    ("fac_metric_collector", {
        "model": "gpt-4.1-nano",
        "provider": "openai"
    }),
]
