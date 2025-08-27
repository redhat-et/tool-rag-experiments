from typing import List

from evaluator.algorithm_factory import Spec

EVALUATION_SPECIFICATION: List[Spec] = [
    ("no_tool_rag_baseline", {}),
    ("basic_tool_rag", {"top_k": 3}),
]
