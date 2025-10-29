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
    reference_model_id: Optional[str] = None

    # The number of queries to include in the evaluation or None to include all available queries.
    queries_num: int | None


class ModelConfig(BaseModel):
    id: str
    url: AnyUrl
    provider_id: ProviderId


class EnvironmentConfig(BaseModel):
    # The ID of the model to be used for inference during the experiment.
    model_id: str


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
