from abc import ABC, abstractmethod
from typing import Dict, ClassVar, List, Any


class MetricCollector(ABC):
    __collector_name__: ClassVar[str | None] = None

    def __init__(self, settings: Dict):
        self._settings = settings
        self.collection_active = False

    @classmethod
    def get_name(cls) -> str:
        return cls.__collector_name__

    def get_unique_id(self) -> str:
        return f"{self.get_name()}:{self._settings}"

    @abstractmethod
    def get_collected_metrics_names(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def set_up(self) -> None:
        self.collection_active = True

    @abstractmethod
    def prepare_for_measurement(self, query: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def register_measurement(self, query: str, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def tear_down(self) -> None:
        self.collection_active = False

    @abstractmethod
    def report_results(self) -> Dict[str, Any] or None:
        if self.collection_active:
            raise RuntimeError(f"Metric collector {self.get_name()}: cannot report results while collection is active")
        return None
