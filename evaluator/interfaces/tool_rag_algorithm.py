from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List, Tuple, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from evaluator.components.data_provider import QuerySpecification

# the algorithm result consists of the query response and the list of retrieved tools, if available
AlgoResponse = Tuple[Dict[str, Any], Union[List[str], None]]


class ToolRagAlgorithm(ABC):
    __algo_name__: ClassVar[str | None] = None

    def __init__(self, settings: Dict):
        self._settings = settings

    @classmethod
    def get_name(cls) -> str:
        return cls.__algo_name__

    def get_unique_id(self) -> str:
        return f"{self.get_name()}:{self._settings}" if self._settings else self.get_name()

    @abstractmethod
    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        raise NotImplementedError()

    @abstractmethod
    def tear_down(self) -> None:
        raise NotImplementedError()
