import os
from typing import List, Dict
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Milvus, VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from pymilvus import connections, utility

from evaluator.components.data_provider import QuerySpecification
from evaluator.eval_spec import VERBOSE
from evaluator.utils.module_extractor import register_tool_rag_algorithm
from evaluator.interfaces.tool_rag_algorithm import ToolRagAlgorithm, AlgoResponse

from dotenv import load_dotenv

load_dotenv()

MILVUS_CONNECTION_ALIAS = "tools_connection"
COLLECTION_NAME = "tools_collection"
OVERRIDE_COLLECTION = True

DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOOL_SELECTION_K = 3


@register_tool_rag_algorithm("basic_tool_rag")
class BasicToolRagAlgorithm(ToolRagAlgorithm):

    model: BaseChatModel or None
    vector_store: VectorStore or None

    def __init__(self, settings: Dict):
        super().__init__(settings)
        self.model = None
        self.vector_store = None

    @staticmethod
    def __tool_to_doc(tool: BaseTool) -> Document:
        schema_str = tool.args_schema.schema_json(indent=2) if getattr(tool, "args_schema", None) else ""
        text = f"{tool.name}\n\n{tool.description}\n\nArgs schema:\n{schema_str}"
        meta = {
            "tool_name": tool.name,
            "json": tool.model_dump_json(),
        }
        return Document(page_content=text, metadata=meta)

    @staticmethod
    def __doc_to_tool(doc: Document) -> BaseTool:
        return BaseTool.model_validate_json(doc["metadata"]["json"])

    @staticmethod
    def __get_or_index_tools(tools: List[BaseTool]) -> VectorStore:
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME",
                                                                DEFAULT_EMBEDDING_MODEL_NAME))
        milvus_uri = os.getenv("MILVUS_PATH")

        connections.connect(alias=MILVUS_CONNECTION_ALIAS, uri=milvus_uri)
        if not OVERRIDE_COLLECTION and utility.has_collection(COLLECTION_NAME):
            print(f"[INFO] Loading Milvus server collection: {COLLECTION_NAME}")
            return Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": milvus_uri}
            )

        print(f"[INFO] Creating new Milvus collection on the server: {COLLECTION_NAME}")
        docs = [Document(page_content=tool.model_dump_json(), metadata=tool.metadata) for tool in tools]
        return Milvus.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": milvus_uri},
            drop_old=True,
        )

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        self.model = model
        self.vector_store = self.__get_or_index_tools(tools)

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        if not self.vector_store:
            raise RuntimeError("process_query called before set_up")

        top_k = self._settings.get("top_k", DEFAULT_TOOL_SELECTION_K)
        relevant_tool_defs = self.vector_store.similarity_search(query_spec.query, k=top_k)
        relevant_tools = [BasicToolRagAlgorithm.__doc_to_tool(d) for d in relevant_tool_defs]

        agent = create_react_agent(self.model, relevant_tools)
        print_mode = "debug" if VERBOSE else ()
        response = await agent.ainvoke(
            input={"messages": query_spec.query},
            max_iterations=6,
            print_mode=print_mode
        )
        return response, [tool.name for tool in relevant_tools]

    def tear_down(self) -> None:
        connections.disconnect(alias=MILVUS_CONNECTION_ALIAS)
