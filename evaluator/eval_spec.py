from typing import List

from evaluator.utils.module_extractor import Spec

EVALUATED_ALGORITHMS: List[Spec] = [
    ("no_tool_rag_baseline", {}),
    #("basic_tool_rag", {"top_k": 3}),
]

METRIC_COLLECTORS: List[Spec] = [
    ("basic_metric_collector", {}),
    ("fac_metric_collector", {
        "show_judge_output": True,        # Show judge model output for each query
        "show_detailed_explanation": True  # Show full judge model response (set to True for debugging)
    }),
]
