from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from evaluator.utils.module_extractor import register_tool_rag_algorithm
from evaluator.interfaces.tool_rag_algorithm import ToolRagAlgorithm


@register_tool_rag_algorithm("no_tool_rag_baseline")
class NoToolRagAlgorithm(ToolRagAlgorithm):
    def __init__(self, settings: Dict):
        super().__init__(settings)
        self.agent = None

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]):
        self.agent = create_react_agent(model, tools)

    async def process_query(self, query: str):
        if not self.agent:
            raise RuntimeError("process_query called before set_up")
        return await self.agent.ainvoke({"messages": query})

    def tear_down(self):
        pass
