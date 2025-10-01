from abc import ABC, abstractmethod
from typing import Dict, ClassVar, List, Any

from evaluator.components.data_provider import QuerySpecification
from evaluator.components.llm_provider import get_llm
from evaluator.config.schema import ModelConfig


class MetricCollector(ABC):
    __collector_name__: ClassVar[str | None] = None

    def __init__(self, settings: Dict, model_config: List[ModelConfig]):
        self._settings = settings
        self._model_config = model_config

        self.collection_active = False

    @classmethod
    def get_name(cls) -> str:
        return cls.__collector_name__

    def get_unique_id(self) -> str:
        return f"{self.get_name()}:{self._settings}"

    def _get_llm(self, model_id: str, **kwargs):
        """ A helper method for subclasses. """
        return get_llm(model_id, self._model_config, **kwargs)

    @abstractmethod
    def get_collected_metrics_names(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def set_up(self) -> None:
        self.collection_active = True

    @abstractmethod
    def prepare_for_measurement(self, query_spec: QuerySpecification) -> None:
        raise NotImplementedError()

    @abstractmethod
    def register_measurement(self, query_spec: QuerySpecification, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def tear_down(self) -> None:
        self.collection_active = False

    @abstractmethod
    def report_results(self) -> Dict[str, Any] or None:
        if self.collection_active:
            raise RuntimeError(f"Metric collector {self.get_name()}: cannot report results while collection is active")
        return None
