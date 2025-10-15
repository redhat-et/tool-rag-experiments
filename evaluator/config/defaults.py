import os

from dotenv import load_dotenv

from evaluator.config.schema import ProviderId

load_dotenv()

VERBOSE = False

DEFAULT_CONFIG = {
    "data": {
        "query_file_paths": [
            "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_category.json",
            "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_instruction.json",
            "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G1_tool.json",
        ],
        "fine_tuning_query_file_paths": [
            "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G2_category.json",
            "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G2_instruction.json",
            "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/refs/heads/master/solvable_queries/test_instruction/G3_instruction.json",
        ],
        "tool_file_paths": [
            "https://huggingface.co/datasets/stabletoolbench/ToolEnv2404/resolve/main/toolenv2404_filtered.tar.gz",
        ],
        "reference_answers_path": "https://huggingface.co/datasets/stabletoolbench/baselines/resolve/main/data_baselines.zip",
        "reference_model_id": "chatgpt_cot",
        "queries_num": 5,
    },

    "models": [
        {"id": "Qwen/Qwen3-8B", "url": os.getenv("QWEN_MODEL_URL"), "provider_id": ProviderId.VLLM},
        {"id": "granite32-8b", "url": os.getenv("GRANITE_MODEL_URL"), "provider_id": ProviderId.VLLM},
        {"id": "llama32-3b", "url": os.getenv("LLAMA_32_MODEL_URL"), "provider_id": ProviderId.VLLM},
        {"id": "meta-llama/Llama-3.1-8B-Instruct", "url": os.getenv("LLAMA_31_MODEL_URL"), "provider_id": ProviderId.VLLM},
        {"id": "AtlaAI/Selene-1-Mini-Llama-3.1-8B", "url": os.getenv("SELENE_JUDGE_MODEL_URL"), "provider_id": ProviderId.VLLM},
    ],

    "environments": [
        {"model_id": "granite32-8b", "irrelevant_tools_ratio": 0.0, "irrelevant_tools_from_same_categories": True},
    ],

    "algorithms": [
        {
            "module_name": "tool_rag",
            "label": "Tool RAG",
            "settings":
                {
                    # basic
                    "top_k": 15,
                    "embedding_model_id": "intfloat/e5-large-v2",  # can also be a local path to a fine-tuned model
                    "similarity_metric": "COSINE",
                    "index_type": "FLAT",
                    "indexed_tool_def_parts": ["name", "description"],

                    # preprocessing
                    "text_preprocessing_operations": None,
                    "max_document_size": 256,

                    # hybrid search
                    "hybrid_mode": False,
                    "analyzer_params": None,
                    "fusion_type": "rrf",
                    "fusion_k": 100,
                    "fusion_alpha": 0.5,

                    # reranking
                    "cross_encoder_model_name": None,  # "BAAI/bge-reranker-large",
                    "reranker_pool_size": 80,

                    # query rewriting / decomposition
                    "enable_query_decomposition": False,
                    "enable_query_rewriting": False,
                    "query_rewriting_model_id": "Qwen/Qwen3-8B",
                    "min_sub_tasks": 1,
                    "max_sub_tasks": 5,
                    "query_rewrite_tool_suggestions_num": 3,

                    # post-retrieval filtering
                    "tau": None,  # 0.3,
                    "sim_threshold": None,  # 0.95,
                }
        },
    ],

    "metric_collectors": [
        {
            "module_name": "answer_quality_metric_collector",
            "settings": dict(
                judges={
                    "binary_task_success_no_ref": "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
                }
            )
        },
        {"module_name": "tool_selection_metric_collector", "settings": {}},
        {"module_name": "tool_retrieval_metric_collector", "settings": {"ks": [1, 3, 5, 10], "ap_rel_threshold": 1.0}},
        {"module_name": "efficiency_metric_collector", "settings": {}},
    ],
}
