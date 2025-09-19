from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from evaluator.components.data_provider import QuerySpecification
from evaluator.eval_spec import VERBOSE
from evaluator.utils.module_extractor import register_tool_rag_algorithm
from evaluator.interfaces.tool_rag_algorithm import ToolRagAlgorithm


@register_tool_rag_algorithm("no_tool_rag_baseline")
class NoToolRagAlgorithm(ToolRagAlgorithm):
    def __init__(self, settings: Dict):
        super().__init__(settings)
        self.model = None
        self.all_tools = None

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]):
        self.model = model
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

    async def process_query(self, query_spec: QuerySpecification):
        if not self.all_tools:
            raise RuntimeError("process_query called before set_up")

        tools = self._filter_relevant_tools(query_spec)
        agent = create_react_agent(self.model, tools)
        print_mode = "debug" if VERBOSE else ()
        return await agent.ainvoke({"messages": query_spec.query}, max_iterations=6, print_mode=print_mode)

    def tear_down(self):
        pass
