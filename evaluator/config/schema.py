from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    AnyUrl,
)


class ProviderId(str, Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # URLs of the files to fetch the tools from.
    # Will only be downloaded if not already available locally.
    # As of now, providing more than one path is not supported!
    tool_file_paths: List[AnyUrl] = []

    # URLs of the files to fetch the queries from.
    # Will only be downloaded if not already available locally.
    query_file_paths: List[AnyUrl] = []

    # URLs of the files to fetch the fine-tuning query dataset from.
    fine_tuning_query_file_paths: Optional[List[AnyUrl]] = None

    # URL of the archive containing the reference answers to the queries.
    reference_answers_path: Optional[AnyUrl] = None

    # the ID of the model that produced the reference answers.
    reference_model_id: str

    # The number of queries to include in the evaluation or None to include all available queries.
    queries_num: int | None


class ModelConfig(BaseModel):
    id: str
    url: AnyUrl
    provider_id: ProviderId


class EnvironmentConfig(BaseModel):
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


class AlgorithmConfig(BaseModel):
    module_name: str
    label: Optional[str] = None
    settings: Dict[str, Any]


class MetricCollectorConfig(BaseModel):
    module_name: str
    settings: Dict[str, Any]


class EvaluationConfig(BaseModel):
    """
    Top-level configuration of the evaluator framework.
    """
    model_config = ConfigDict(extra="forbid")

    data: DatasetConfig
    models: List[ModelConfig]
    environments: List[EnvironmentConfig]
    algorithms: List[AlgorithmConfig]
    metric_collectors: List[MetricCollectorConfig]
