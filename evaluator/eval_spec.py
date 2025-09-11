from typing import List, Dict, Any

from evaluator.utils.module_extractor import Spec

EVALUATED_ALGORITHMS: List[Spec] = [
    ("no_tool_rag_baseline", {}),
    #("basic_tool_rag", {"top_k": 3}),
]

METRIC_COLLECTORS: List[Spec] = [
    ("basic_metric_collector", {}),
    ("fac_metric_collector", {
        "verbose": True  # True for detailed output, False for minimal output
    }),
]

DATASET_SETTINGS: Dict[str, Any] = {
    # URLs of the files to fetch the queries from.
    # Will only be downloaded if not already available locally.
    "query_files": [
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_category.json",
        #"https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_instruction.json",
        #"https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_tool.json",
    ],

    # URLs of the files to fetch the tools from.
    # Will only be downloaded if not already available locally.
    # TODO: as of now, providing more than one path is not supported!
    "tool_files": [
        "https://huggingface.co/datasets/stabletoolbench/ToolEnv2404/resolve/main/toolenv2404_filtered.tar.gz",
    ],

    # Tool categories (according to the ToolBench dataset specification) to include in the evaluation.
    # Set this to None to include all categories.
    "tool_categories": ["Weather"],

    # The maximal number of tools to include in the evaluation.
    # The actual number will be smaller than this value if the selected categories do not contain enough tools.
    "max_tools_num": 10,

    # The maximal number of queries to include in the evaluation.
    # The actual number will be smaller than this value if not enough queries in the dataset
    # use the tools selected for evaluation.
    "max_queries_num": 10,
}
