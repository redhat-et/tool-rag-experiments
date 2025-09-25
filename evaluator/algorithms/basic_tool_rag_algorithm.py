import os
from typing import List, Dict
from langchain.docstore.document import Document
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_milvus import Milvus
from langgraph.prebuilt import create_react_agent
from pymilvus import connections, utility

from evaluator.components.data_provider import QuerySpecification
from evaluator.eval_spec import VERBOSE
from evaluator.utils.module_extractor import register_tool_rag_algorithm
from evaluator.interfaces.tool_rag_algorithm import ToolRagAlgorithm, AlgoResponse

from dotenv import load_dotenv

from evaluator.utils.utils import print_verbose

load_dotenv()

MILVUS_CONNECTION_ALIAS = "tools_connection"
COLLECTION_NAME = "tools_collection"
OVERRIDE_COLLECTION = True

DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOOL_SELECTION_K = 3

if not VERBOSE:
    # Silence gRPC C-core and tracing
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@register_tool_rag_algorithm("basic_tool_rag")
class BasicToolRagAlgorithm(ToolRagAlgorithm):

    model: BaseChatModel or None
    vector_store: VectorStore or None

    # due to the limitations of Langchain tools we cannot truly serialize them. Therefore, indexing
    # the tools themselves is not possible. Instead, we keep all tools in memory and only index their unique IDs (names)
    # which are later used to retrieve the actual tools.
    tool_name_to_base_tool: Dict[str, BaseTool] or None

    def __init__(self, settings: Dict):
        super().__init__(settings)
        self.model = None
        self.tool_name_to_base_tool = None
        self.vector_store = None

    @staticmethod
    def _create_docs_from_tools(tools: List[BaseTool]) -> List[Document]:
        documents = []
        for tool in tools:
            documents.append(Document(page_content=tool.description, metadata={"name": tool.name}))
        return documents

    def _index_tools(self, tools: List[BaseTool]) -> None:
        self.tool_name_to_base_tool = {tool.name: tool for tool in tools}

        embedding_model_name = self._settings.get("embedding_model_id", DEFAULT_EMBEDDING_MODEL_NAME)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        milvus_uri = os.getenv("MILVUS_PATH")

        connections.connect(alias=MILVUS_CONNECTION_ALIAS, uri=milvus_uri)
        if not OVERRIDE_COLLECTION and utility.has_collection(COLLECTION_NAME):
            print_verbose(f"Loading Milvus server collection: {COLLECTION_NAME}")
            self.vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": milvus_uri}
            )

        print_verbose(f"Creating new Milvus collection on the server: {COLLECTION_NAME}")
        self.vector_store = Milvus.from_documents(
            documents=self._create_docs_from_tools(tools),
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": milvus_uri},
            drop_old=True,
        )

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        self.model = model
        self._index_tools(tools)

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        if not self.vector_store:
            raise RuntimeError("process_query called before set_up")

        top_k = self._settings.get("top_k", DEFAULT_TOOL_SELECTION_K)
        relevant_documents = self.vector_store.similarity_search(query_spec.query, k=top_k)
        relevant_tool_names = [d.metadata["name"] for d in relevant_documents]
        print_verbose(f"Retrieved tools for query #{query_spec.id}: {relevant_tool_names}")
        relevant_tools = [self.tool_name_to_base_tool[name] for name in relevant_tool_names]

        agent = create_react_agent(self.model, relevant_tools)
        print_mode = "debug" if VERBOSE else ()
        response = await agent.ainvoke(
            input={"messages": query_spec.query},
            max_iterations=6,
            print_mode=print_mode
        )
        return response, relevant_tool_names

    def tear_down(self) -> None:
        connections.disconnect(alias=MILVUS_CONNECTION_ALIAS)
