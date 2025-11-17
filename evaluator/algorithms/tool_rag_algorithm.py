import json
import math
import os
from itertools import chain
from typing import List, Dict, Tuple, Any

from langchain.docstore.document import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from pymilvus import connections, utility

from evaluator.components.data_provider import QuerySpecification
from evaluator.components.llm_provider import query_llm
from evaluator.config.schema import ModelConfig
from evaluator.utils.module_extractor import register_algorithm
from evaluator.interfaces.algorithm import AlgoResponse
from evaluator.utils.utils import log_verbose
from evaluator.algorithms.base_retrieval_algorithm import BaseRetrievalAlgorithm

from dotenv import load_dotenv

load_dotenv()

MILVUS_CONNECTION_ALIAS = "tools_connection"
DEFAULT_RECURSION_LIMIT = 10  # Tool RAG needs fewer iterations since tools are available upfront
DEFAULT_MAX_ITERATIONS = 6  # Maximum iterations for agent execution

@register_algorithm("tool_rag")
class ToolRagAlgorithm(BaseRetrievalAlgorithm):
    """
    Tool RAG algorithm with optional query decomposition and rewriting.
    
    Optional configurable settings (to be provided in the 'settings' dictionary):

    - top_k: the number of documents to retrieve on a given query.
    - embedding_model_id: the ID of the model to use for embedding calculation.
    - similarity_metric: the metric to use for embedding distance calculation - can be COSINE, L2 or IP (inner product).
    - index_type: the type of Milvus index to build - can be FLAT, HNSW, IVF_FLAT or any other supported type.
    - tau: an optional threshold between 0 and 1 specifying the maximal allowed distance to the query. All tools having
      greater distance will be filtered. Set to None to disable this filter.
    - sim_threshold: an optional threshold between 0 and 1 specifying the maximal allowed similarity between the retrieved
      tools. Sets of highly similar tools will be filtered to only retain one. Set to None to disable this filter.
    - text_preprocessing_operations: a list of preprocessing operations to apply on the indexed documents and the
      query text. The following operations are supported: 'unicode_normalization', 'lowercase', 'collapse_whitespaces',
      'split_camel_snake_case'. Set this parameter to None to disable preprocessing or to 'all' to enable all operations.
    - max_document_size: the maximal size, in characters, of a single indexed document, or None to disable the size limit.
    - indexed_tool_def_parts: the parts of the MCP tool definition to be used for index construction, such as 'name',
      'description', 'args', etc.
      You can also include 'examples' (or 'examples') to append example queries for each tool if provided
      via the 'examples' setting (see defaults below).
    - hybrid_mode: True to enable hybrid (sparse + dense) search and False to only enable dense search.
    - analyzer_params: parameters for the Milvus BM25 analyzer.
    - fusion_type: the algorithm for combining the dense and the sparse scores if hybrid mode is activated. Milvus only
      supports "weighted" and "rrf".
    - fusion_alpha: the relative weight of the dense retriever score. The final score is calculated as
      alpha*dense + (1-alpha)*sparse. This parameter is only used with the "weighted" fusion type.
    - fusion_k: the k value to use for the "rrf" hybrid fusion mode.
    - cross_encoder_model_name: the name of the model to use for reranking or None to disable reranking.
    - reranker_pool_size: the number of results to retrieve from the vector DB before reranking. Must be greater than or
      equal to top_k.
    - enable_query_decomposition: True to enable query decomposition into subtasks and False otherwise.
    - enable_query_rewriting: True to enable (sub-)query rewriting into a list of relevant APIs and False otherwise.
    - query_rewriting_model_id: the ID of the model to use for query rewriting and decomposition.
    - min_sub_tasks: the minimum number of tasks to decompose the original query into.
    - max_sub_tasks: the maximum number of tasks to decompose the original query into.
    - query_rewrite_tool_suggestions_num: the maximal number of tool APIs to produce from the original query during rewriting.
    - recursion_limit: Maximum iterations for LangGraph agent (default: 10).
    - max_iterations: Maximum iterations for agent execution (default: 6).
    """

    query_rewriting_model: BaseChatModel or None

    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        super().__init__(settings, model_config, label)
        self.query_rewriting_model = None

    def get_default_settings(self) -> Dict[str, Any]:
        """Extend base settings with ToolRAG-specific settings."""
        base_settings = super().get_default_settings()
        base_settings.update({
            # ToolRAG specific
            "top_k": 10,
            "collection_name": "tools_collection",
            "drop_old_collection": True,
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
            
            # Query rewriting / decomposition
            "enable_query_decomposition": False,
            "enable_query_rewriting": False,
            "query_rewriting_model_id": "llama32-3b",
            "min_sub_tasks": 1,
            "max_sub_tasks": 5,
            "query_rewrite_tool_suggestions_num": 3,
        })
        return base_settings

    def _index_tools(self, tools: List[BaseTool]) -> None:
        """Index tools into Milvus vector store."""
        self._update_tool_mapping(tools)

        milvus_uri = os.getenv("MILVUS_URL")
        connections.connect(alias=MILVUS_CONNECTION_ALIAS, uri=milvus_uri)
        
        collection_name = self._settings.get("collection_name", "tools_collection")
        drop_old = bool(self._settings.get("drop_old_collection", True))
        
        if not drop_old and utility.has_collection(collection_name):
            log_verbose(f"Loading Milvus server collection: {collection_name}")
            # Load existing collection - handled by base class if needed
            # For now, we rebuild
        
        log_verbose(f"Creating new Milvus collection on the server: {collection_name}")
        self._build_vector_store(tools, collection_name, drop_old=drop_old)

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        super().set_up(model, tools)

        # Initialize reranker
        self._initialize_reranker(self._settings["top_k"])

        # Initialize query rewriting model
        if self._settings["enable_query_decomposition"] or self._settings["enable_query_rewriting"]:
            self.query_rewriting_model = self._get_llm(self._settings["query_rewriting_model_id"])

        # Build vector index
        self._index_tools(tools)

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decomposes a given, possibly highly complex query into a list of simpler subtasks.
        """
        system_prompt = "You are a helpful assistant."

        user_prompt = (
            "Split the USER REQUEST provided below into atomic, simple subtasks. "
            "Each subtask should map to a single API/tool call. "
            f"Produce between {self._settings['min_sub_tasks']} and {self._settings['max_sub_tasks']} subtasks. "
            f"Subtasks should be written in imperative style and contain at most 25 words."
            f"Merge near-duplicates. "
            f"Skip steps that do not require a tool. "
            f"Return JSON: {{\"subtasks\": [\"...\"]}}"
            f"USER REQUEST:\n {query}\n"
        )

        raw = query_llm(self.query_rewriting_model, system_prompt, user_prompt)
        parsed = self._safe_json_parse(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("subtasks"), list):
            items = [str(x) for x in parsed["subtasks"]]
        else:
            # fallback to line-by-line if JSON wasn't returned
            items = self._lines(raw)

        steps: List[str] = []
        for s in items:
            s = self._strip_numbering(s)
            if not s:
                continue
            steps.append(s)
            if len(steps) >= self._settings['max_sub_tasks']:
                break

        steps = self._dedup_keep_order(steps)
        if not steps:
            steps = [query]

        return steps

    def _rewrite_query_to_tool_descriptions(self, query: str) -> List[str]:
        """
        Transforms a query into a list of hypothetical tool APIs required to complete it.
        """
        system_prompt = "You are a helpful assistant."

        num_tools = self._settings['query_rewrite_tool_suggestions_num']
        user_prompt = (
            f"Generate short descriptions of API(s) that can be used to address USER REQUEST as specified below. "
            f"Each description must be concise and phrased like a capability (not an answer). "
            f"Do NOT answer the user. "
            f"Write up to {num_tools} short descriptions. "
            f"Each description should contain at most 25 words. "
            f"Output one description per line."
            f"USER REQUEST:\n {query}\n"
        )

        raw = query_llm(self.query_rewriting_model, system_prompt, user_prompt)
        items = self._lines(raw)[:max(1, num_tools)]
        items = self._dedup_keep_order([s for s in items if s])

        return items

    @staticmethod
    def _merge_tool_docs(
            tool_docs_with_scores_lists: List[List[Tuple[Document, float]]]) -> List[Tuple[Document, float]]:
        """Merge multiple lists of tool documents, keeping highest scores."""
        merged = []

        for doc, score in chain.from_iterable(tool_docs_with_scores_lists):
            # Find if an equal document already exists in the merged list
            idx = next((i for i, (d, _) in enumerate(merged) if d.metadata["name"] == doc.metadata["name"]), None)

            if idx is None:
                merged.append((doc, score))
            else:
                # Keep the higher score
                if score > merged[idx][1]:
                    merged[idx] = (doc, score)

        merged.sort(key=lambda pair: pair[1], reverse=True)
        return merged

    async def _get_tools_for_sub_query(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Get tools for a single sub-query, optionally with query rewriting."""
        if not self._settings["enable_query_rewriting"]:
            # just fetch the tools from the DB based on the query text
            return await self._fetch_tools_for_sub_query_from_vector_db(query, k)

        tool_descriptions = self._rewrite_query_to_tool_descriptions(query)
        tool_candidates_per_tool_descriptions = math.ceil(max(k / len(tool_descriptions), 1))
        all_tool_lists = []
        for tool_description in tool_descriptions:
            current_tools = await self._fetch_tools_for_sub_query_from_vector_db(
                tool_description,
                tool_candidates_per_tool_descriptions,
            )
            all_tool_lists.append(current_tools)

        return self._merge_tool_docs(all_tool_lists)

    async def _fetch_tools_for_sub_query_from_vector_db(self, sub_query: str, k: int) -> List[Tuple[Document, float]]:
        """Fetch tools directly from vector DB for a sub-query."""
        query_text = self._preprocess_text(sub_query)

        return self._vector_search_with_scores(query_text, k)

    async def _get_tools_for_query(self, query: str) -> List[Tuple[Document, float]]:
        """Get tools for the main query, handling decomposition if enabled."""
        # Use base class helper to determine k (accounts for reranking pool size)
        total_k = self._get_retrieval_k(self._settings["top_k"])

        if not self._settings["enable_query_decomposition"]:
            # there is just one sub-query
            return await self._get_tools_for_sub_query(query, total_k)

        sub_queries = self._decompose_query(query)
        tools_per_sub_query = math.ceil(max(total_k / len(sub_queries), 1))
        all_tool_lists = []
        for sub_q in sub_queries:
            current_tools = await self._get_tools_for_sub_query(sub_q, tools_per_sub_query)
            all_tool_lists.append(current_tools)

        return self._merge_tool_docs(all_tool_lists)

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        if not self.vector_store:
            raise RuntimeError("process_query called before set_up")

        log_verbose(f"Retrieving documents for query: {query_spec.query}")
        docs_and_scores = await self._get_tools_for_query(query_spec.query)

        relevant_documents = self._postprocess_results(
            docs_and_scores,
            self._preprocess_text(query_spec.query),
            self._settings.get("top_k"),
        )
        
        # Use base class helper to safely convert docs to tools (fixes KeyError bug)
        relevant_tools = self._docs_to_tools(relevant_documents)
        relevant_tool_names = [tool.name for tool in relevant_tools]

        log_verbose(f"Retrieved tools for query #{query_spec.id}: {relevant_tool_names}")

        agent = create_react_agent(self._model, relevant_tools)
        return await self._invoke_agent_on_query(
            agent,
            query_spec.query,
            config={
                "recursion_limit": self._settings.get("recursion_limit", DEFAULT_RECURSION_LIMIT),
                "max_iterations": self._settings.get("max_iterations", DEFAULT_MAX_ITERATIONS)
            }
        ), relevant_tool_names

    def tear_down(self) -> None:
        connections.disconnect(alias=MILVUS_CONNECTION_ALIAS)
