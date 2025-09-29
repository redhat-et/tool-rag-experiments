import os
import re
import unicodedata

import numpy as np

from typing import List, Dict, Tuple, Any
from langchain.docstore.document import Document
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

_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_WS_RE = re.compile(r'\s+')


MILVUS_CONNECTION_ALIAS = "tools_connection"
COLLECTION_NAME = "tools_collection"
OVERRIDE_COLLECTION = True


DEFAULT_SETTINGS = {
    "top_k": 3,
    "embedding_model_id": "all-MiniLM-L6-v2",
    "similarity_metric": "COSINE",
    "index_type": "FLAT",
    "tau": None,
    "text_preprocessing_operations": None,
    "max_document_size": None,
    "indexed_tool_def_parts": ["name", "description"],
}

if not VERBOSE:
    # Silence gRPC C-core and tracing
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class L2Wrapper:
    def __init__(self, base):
        self.base = base

    def embed_documents(self, texts):
        x = np.array(self.base.embed_documents(texts), dtype=np.float32)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return x.tolist()

    def embed_query(self, text):
        x = np.array(self.base.embed_query(text), dtype=np.float32)
        x /= (np.linalg.norm(x) + 1e-12)
        return x.tolist()


@register_tool_rag_algorithm("basic_tool_rag")
class BasicToolRagAlgorithm(ToolRagAlgorithm):
    """
    Optional configurable settings (to be provided in the 'settings' dictionary):

    - top_k: the number of documents to retrieve on a given query.
    - embedding_model_id: the ID of a model to use for embedding calculation.
    - similarity_metric: the metric to use for embedding distance calculation - can be COSINE, L2 or IP (inner product).
    - index_type: the type of Milvus index to build - can be FLAT, HNSW, IVF_FLAT or any other supported type.
    - tau: an optional threshold between 0 and 1 to filter the retrieved documents post-search (None to disable filtering).
    - text_preprocessing_operations: a list of preprocessing operations to apply on the indexed documents and the
      query text. The following operations are supported: 'unicode_normalization', 'lowercase', 'collapse_whitespaces',
      'split_camel_snake_case'. Set this parameter to None to disable preprocessing or to 'all' to enable all operations.
    - max_document_size: the maximal size, in characters, of a single indexed document, or None to disable the size limit.
    - indexed_tool_def_parts: the parts of the MCP tool definition to be used for index construction, such as 'name',
      'description', 'args', etc.
    """

    model: BaseChatModel or None
    vector_store: Milvus or None

    # due to the limitations of Langchain tools we cannot truly serialize them. Therefore, indexing
    # the tools themselves is not possible. Instead, we keep all tools in memory and only index their unique IDs (names)
    # which are later used to retrieve the actual tools.
    tool_name_to_base_tool: Dict[str, BaseTool] or None

    def __init__(self, settings: Dict, embedding_function: HuggingFaceEmbeddings):
        super().__init__(settings)
        for key, value in DEFAULT_SETTINGS.items():
            self._settings.setdefault(key, value)

        self.model = None
        self.tool_name_to_base_tool = None
        self.vector_store = None
        self.embeddings_function = embedding_function

    def _preprocess_text(self, text: str) -> str:
        ops = self._settings["text_preprocessing_operations"]
        if not ops:
            return text

        t = text
        if "unicode_normalization" in ops or "all" in ops:
            t = unicodedata.normalize('NFC', t)
        if "lowercase" in ops or "all" in ops:
            t = t.lower()
        if "collapse_whitespaces" in ops or "all" in ops:
            t = _WS_RE.sub(' ', t).strip()
        if "split_camel_snake_case" in ops or "all" in ops:
            t = t.replace('_', ' ')  # snake_case -> snake case
            t = _CAMEL_RE.sub(' ', t)  # camelCase -> camel Case

        return t

    def _truncate(self, s: str, sep: str = " ... ") -> str:
        max_chars = self._settings["max_document_size"]
        if max_chars is None or len(s) <= max_chars:
            return s

        # Performs a simple head-tail truncation to max_chars characters.
        # We may consider supporting more sophisticated techniques, such as token-based truncation in the future.
        half = (max_chars - len(sep)) // 2
        return s[:half] + sep + s[-half:]

    @staticmethod
    def _render_args_schema(schema: Dict[str, Any]) -> str:
        if not schema:
            return ""
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        ordered = list(required) + [p for p in props.keys() if p not in required]
        # Keep it terse: "name:type" when available
        parts = []
        for name in ordered:
            spec = props.get(name) or {}
            t = spec.get("type")
            if isinstance(t, str):
                parts.append(f"{name}:{t}")
            else:
                parts.append(name)
        return " ".join(parts)

    @staticmethod
    def _render_examples(examples: List[str], max_examples: int = 3) -> str:
        exs = (examples or [])[:max_examples]
        return " || ".join(exs)

    def _compose_tool_text(self, tool: BaseTool) -> str:
        parts_to_include = self._settings["indexed_tool_def_parts"]
        if not parts_to_include:
            raise ValueError("indexed_tool_def_parts must be a non-empty list")

        segments = []
        for p in parts_to_include:
            if p.lower() == "name":
                val = tool.name
                if val:
                    segments.append(f"name: {val}")
            elif p.lower() == "description":
                val = tool.description
                if val:
                    segments.append(f"desc: {val}")
            elif p.lower() == "args":
                val = self._render_args_schema(tool.args_schema)
                if val:
                    segments.append(f"args: {val}")
            elif p.lower() == "tags":
                tags = tool.tags or []
                if tags:
                    segments.append(f"tags: {' '.join(tags)}")

        if not segments:
            raise ValueError(f"The following tool contains none of the fields listed in indexed_tool_def_parts:\n{tool}")
        text = " | ".join(segments)

        # one-pass preprocess + truncation
        text = self._preprocess_text(text)
        text = self._truncate(text)
        return text

    def _create_docs_from_tools(self, tools: List[BaseTool]) -> List[Document]:
        documents = []
        for tool in tools:
            page_content = self._compose_tool_text(tool)
            documents.append(Document(page_content=page_content, metadata={"name": tool.name}))
        return documents

    def _index_tools(self, tools: List[BaseTool]) -> None:
        self.tool_name_to_base_tool = {tool.name: tool for tool in tools}

        embeddings = HuggingFaceEmbeddings(model_name=self._settings["embedding_model_id"])
        if self._settings["similarity_metric"] == "COSINE":
            # L2-normalizing embedding vectors before cosine similarity makes the results more stable
            embeddings = L2Wrapper(embeddings)
        milvus_uri = os.getenv("MILVUS_PATH")

        index_params = {
            "index_type": self._settings["index_type"],
            "metric_type": self._settings["similarity_metric"],
        }

        search_params = {
            "metric_type": self._settings["similarity_metric"],
        }

        connections.connect(alias=MILVUS_CONNECTION_ALIAS, uri=milvus_uri)
        if not OVERRIDE_COLLECTION and utility.has_collection(COLLECTION_NAME):
            print_verbose(f"Loading Milvus server collection: {COLLECTION_NAME}")
            self.vector_store = Milvus(
                embedding_function=self.embeddings_function,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": milvus_uri},
                index_params=index_params,
                search_params=search_params,
            )

        print_verbose(f"Creating new Milvus collection on the server: {COLLECTION_NAME}")
        self.vector_store = Milvus.from_documents(
            documents=self._create_docs_from_tools(tools),
            embedding=self.embeddings_function,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": milvus_uri},
            drop_old=True,
            index_params=index_params,
            search_params=search_params,
        )

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        self.model = model
        self._index_tools(tools)
        # self.embeddings_function = embedding_function

    def _threshold_results(self, docs_and_scores: List[Tuple[Document, float]]) -> List[Document]:
        """
        Filters search results according to the tau parameter to only retain those similar enough to the query.
        """
        tau = self._settings["tau"]
        if tau is None:
            # tau not specified - skip post-search filtering
            return [d for (d, s) in docs_and_scores]

        metric = self._settings["similarity_metric"]

        if metric.upper() in {"COSINE", "IP"}:
            kept = [(d, s) for (d, s) in docs_and_scores if s >= tau]
            kept.sort(key=lambda x: x[1], reverse=True)  # higher is better
        elif metric.upper() == "L2":
            kept = [(d, s) for (d, s) in docs_and_scores if s <= tau]
            kept.sort(key=lambda x: x[1])  # lower is better
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return [d for (d, s) in kept]

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        if not self.vector_store:
            raise RuntimeError("process_query called before set_up")

        print_verbose(f"Retrieving documents for query: {query_spec.query}")
        query_text = self._preprocess_text(query_spec.query)

        docs_and_scores = self.vector_store.similarity_search_with_score(query_text, k=self._settings["top_k"])
        relevant_documents = self._threshold_results(docs_and_scores)
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
