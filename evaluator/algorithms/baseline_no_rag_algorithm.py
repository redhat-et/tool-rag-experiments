from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from evaluator.components.data_provider import QuerySpecification
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

    def _filter_relevant_tools(self, query_spec: QuerySpecification) -> List[BaseTool]:
        filtered_tools = []
        for tool in self.all_tools:
            if tool.name in query_spec.golden_tools or tool.name in query_spec.additional_tools:
                filtered_tools.append(tool)
        return filtered_tools

    async def process_query(self, query_spec: QuerySpecification):
        if not self.all_tools:
            raise RuntimeError("process_query called before set_up")

        agent = create_react_agent(self.model, self._filter_relevant_tools(query_spec))
        return await agent.ainvoke({"messages": query_spec.query})

    def tear_down(self):
        pass
