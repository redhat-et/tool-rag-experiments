from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from evaluator.components.data_provider import QuerySpecification


class ToolRagAlgorithm(ABC):
    __algo_name__: ClassVar[str | None] = None

    def __init__(self, settings: Dict):
        self._settings = settings

    @classmethod
    def get_name(cls) -> str:
        return cls.__algo_name__

    def get_unique_id(self) -> str:
        return f"{self.get_name()}:{self._settings}"

    @abstractmethod
    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def process_query(self, query_spec: QuerySpecification) -> dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def tear_down(self) -> None:
        raise NotImplementedError()
