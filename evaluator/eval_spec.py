from typing import List, Dict, Any, Tuple

Spec = Tuple[str, Dict[str, Any]]

VERBOSE = False

EVALUATED_ALGORITHMS: List[Spec] = [
    ("no_tool_rag_baseline", {}),
    #("basic_tool_rag", {"top_k": 5}),
]

METRIC_COLLECTORS: List[Spec] = [
    # ("basic_metric_collector", {}),
    ("fac_metric_collector", {}),
    ("tool_selection_metric_collector", {}),
    #("tool_retrieval_metric_collector", {"ks": [1, 3, 5], "ap_rel_threshold": 1.0}),
]

DATASET_SETTINGS: Dict[str, Any] = {
    # URLs of the files to fetch the queries from.
    # Will only be downloaded if not already available locally.
    "query_files": [
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_category.json",
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_instruction.json",
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_tool.json",
    ],

    # URLs of the files to fetch the tools from.
    # Will only be downloaded if not already available locally.
    # TODO: as of now, providing more than one path is not supported!
    "tool_files": [
        "https://huggingface.co/datasets/stabletoolbench/ToolEnv2404/resolve/main/toolenv2404_filtered.tar.gz",
    ],

    # The number of queries to include in the evaluation or None to include all available queries.
    "queries_num": None,

    # The ratio of relevant to irrelevant tools in the prompt that uses no tool RAG.
    # For instance:
    # - if this value is 0.0, the prompt will only include the correct tools with no irrelevant ones
    # - if this value is 1.0, the prompt will include one irrelevant tool for each relevant tools, i.e., the total
    #   number of tools will be double the number of the correct tools
    # - if this value is 0.5, the prompt will include one irrelevant tool for each two relevant tools (rounding up)
    # Negative values are not allowed.
    "irrelevant_tools_ratio": 0.0,

    # True to fetch irrelevant tools from the same categories as the relevant tools and False to include
    # fully random tools instead. We expect irrelevant tools from the same categories to confuse the model more.
    "irrelevant_tools_from_same_categories": True,
}
