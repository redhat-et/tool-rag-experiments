import random
from typing import Dict, List, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from evaluator.components.data_provider import QuerySpecification
from evaluator.config.schema import ModelConfig
from evaluator.utils.module_extractor import register_algorithm
from evaluator.interfaces.algorithm import Algorithm, AlgoResponse
from evaluator.utils.utils import log_verbose


@register_algorithm("baseline_algorithm")
class BaselineAlgorithm(Algorithm):
    """
    Optional configurable settings (to be provided in the 'settings' dictionary):

    - available_tools_per_query: this number specifies how many tools the model will see per query. Set this
      parameter to None to disable adding irrelevant tools.
    """
    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        super().__init__(settings, model_config, label)
        self.all_tools = None

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        super().set_up(model, tools)
        self.all_tools = tools

    def _filter_relevant_tools(self, query_spec: QuerySpecification) -> List[BaseTool]:
        if query_spec.demo_mode:
            return self.all_tools

        filtered_tools = []
        for tool in self.all_tools:
            if tool.name in query_spec.golden_tools:
                filtered_tools.append(tool)

        if not self._settings["available_tools_per_query"]:
            # tool list augmentation disabled - only return the 'golden' tools
            return filtered_tools

        additional_tools_num = self._settings["available_tools_per_query"] - len(filtered_tools)
        remaining_tools = [t for t in self.all_tools if t not in filtered_tools]
        additional_tools = random.sample(remaining_tools, additional_tools_num)
        filtered_tools.extend(additional_tools)
        random.shuffle(filtered_tools)

        return filtered_tools

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        if not self.all_tools:
            raise RuntimeError("process_query called before set_up")

        tools = self._filter_relevant_tools(query_spec)
        log_verbose(f"The following tools were loaded for query {query_spec.id}:\n{[t.name for t in tools]}")
        agent = create_react_agent(self._model, tools)
        return await self._invoke_agent_on_query(agent, query_spec.query), None

    def tear_down(self) -> None:
        pass

    def get_default_settings(self) -> Dict[str, Any]:
        return {}
