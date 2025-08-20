from typing import Dict
from langgraph.prebuilt import create_react_agent

from evaluator.algorithm_factory import register_tool_rag_algorithm
from evaluator.tool_rag_algorithm import ToolRagAlgorithm


@register_tool_rag_algorithm("no_tool_rag_baseline")
class BM25Algorithm(ToolRagAlgorithm):
    def __init__(self, settings: Dict):
        super().__init__(settings)
        self.__agent = None

    def set_up(self, model, tools):
        self.__agent = create_react_agent(model, tools)

    async def process_query(self, query: str):
        return await self.__agent.ainvoke({"messages": query})

    def tear_down(self):
        pass
