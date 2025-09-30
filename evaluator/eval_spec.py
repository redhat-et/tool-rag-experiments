import os
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class EvaluationEnvSpec(BaseModel):
    # The ID of the model to be used for inference during the experiment.
    model_id: str

    # The ratio of relevant to irrelevant tools in the prompt that uses no tool RAG.
    # For instance:
    # - if this value is 0.0, the prompt will only include the correct tools with no irrelevant ones
    # - if this value is 1.0, the prompt will include one irrelevant tool for each relevant tools, i.e., the total
    #   number of tools will be double the number of the correct tools
    # - if this value is 0.5, the prompt will include one irrelevant tool for each two relevant tools (rounding up)
    # Negative values are not allowed.
    irrelevant_tools_ratio: float

    # True to fetch irrelevant tools from the same categories as the relevant tools and False to include
    # fully random tools instead. We expect irrelevant tools from the same categories to confuse the model more.
    irrelevant_tools_from_same_categories: bool


PluginConfigSpec = Tuple[str, Dict[str, Any]]

VERBOSE = False

EVALUATED_ALGORITHMS: List[PluginConfigSpec] = [
    # ("no_tool_rag_baseline", {}),
    ("tool_rag", {"top_k": 3, "embedding_model_id": "all-MiniLM-L6-v2"}),
]

EXPERIMENTAL_ENVIRONMENT_SETTINGS: List[EvaluationEnvSpec] = [
    EvaluationEnvSpec(
        model_id="Qwen/Qwen3-8B",
        irrelevant_tools_ratio=0.0,
        irrelevant_tools_from_same_categories=True
    ),
]

METRIC_COLLECTORS: List[PluginConfigSpec] = [
    ("answer_quality_metric_collector", dict(judges={
        # "task_success_no_ref": "llama32-3b",
        "task_success_with_ref": "llama32-3b",
    })),
    ("fac_metric_collector", {}),
    ("tool_selection_metric_collector", {}),
    ("tool_retrieval_metric_collector", {"ks": [1, 3, 5], "ap_rel_threshold": 1.0}),
    ("efficiency_metric_collector", {}),
]

DATASET_SETTINGS: Dict[str, Any] = {
    # URLs of the files to fetch the queries from.
    # Will only be downloaded if not already available locally.
    "query_files": [
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_category.json",
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_instruction.json",
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_tool.json",
    ],

    # URLs of the files to fetch the fine-tuning query dataset from.
    "fine_tuning_query_files": [
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G2_category.json",
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G2_instruction.json",
        "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G3_instruction.json",
    ],

    # URLs of the files to fetch the tools from.
    # Will only be downloaded if not already available locally.
    # As of now, providing more than one path is not supported!
    "tool_files": [
        "https://huggingface.co/datasets/stabletoolbench/ToolEnv2404/resolve/main/toolenv2404_filtered.tar.gz",
    ],

    # URL of the archive containing the reference answers to the queries.
    "reference_answers_path": "https://huggingface.co/datasets/stabletoolbench/baselines/resolve/main/data_baselines.zip",

    # the ID of the model that produced the reference answers.
    "reference_model_id": "chatgpt_cot",

    # The number of queries to include in the evaluation or None to include all available queries.
    "queries_num": None,
}

MODEL_ID_TO_URL = {
    "meta-llama/Llama-3.1-8B-Instruct": os.getenv("LLAMA_31_MODEL_URL"),
    "Qwen/Qwen3-8B": os.getenv("QWEN_MODEL_URL"),
    "granite32-8b": os.getenv("GRANITE_MODEL_URL"),
    "llama32-3b": os.getenv("LLAMA_32_MODEL_URL"),
    "AtlaAI/Selene-1-Mini-Llama-3.1-8B": os.getenv("SELENE_JUDGE_MODEL_URL"),
    "llama3.1:8b-instruct-fp16": os.getenv("LLAMA_31_OLLAMA_URL"),
    # more models to be added if needed
}

MODEL_ID_TO_PROVIDER_TYPE = {
    "meta-llama/Llama-3.1-8B-Instruct": "vllm",
    "Qwen/Qwen3-8B": "vllm",
    "granite32-8b": "vllm",
    "llama32-3b": "vllm",
    "AtlaAI/Selene-1-Mini-Llama-3.1-8B": "vllm",
    "llama3.1:8b-instruct-fp16": "ollama",
    # more models to be added if needed
}
