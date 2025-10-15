from typing import Dict, List, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from evaluator.components.data_provider import QuerySpecification
from evaluator.config.schema import ModelConfig
from evaluator.utils.module_extractor import register_algorithm
from evaluator.interfaces.algorithm import Algorithm, AlgoResponse


@register_algorithm("baseline_algorithm")
class BaselineAlgorithm(Algorithm):
    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        super().__init__(settings, model_config, label)
        self.all_tools = None

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        super().set_up(model, tools)
        self.all_tools = tools

    @staticmethod
    def _is_tool_relevant(query_spec: QuerySpecification, tool_name: str) -> bool:
        if tool_name in query_spec.golden_tools:
            return True
        if not query_spec.additional_tools:
            return False
        return tool_name in query_spec.additional_tools

    def _filter_relevant_tools(self, query_spec: QuerySpecification) -> List[BaseTool]:
        filtered_tools = []
        for tool in self.all_tools:
            if self._is_tool_relevant(query_spec, tool.name):
                filtered_tools.append(tool)
        return filtered_tools

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        if not self.all_tools:
            raise RuntimeError("process_query called before set_up")

        tools = self._filter_relevant_tools(query_spec)
        agent = create_react_agent(self._model, tools)
        return await self._invoke_agent_on_query(agent, query_spec.query), None

    def tear_down(self) -> None:
        pass

    def get_default_settings(self) -> Dict[str, Any]:
        return {}
