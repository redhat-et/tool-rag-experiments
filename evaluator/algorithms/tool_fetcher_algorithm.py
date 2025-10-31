from typing import Dict, List, Any, Optional
import json
import os
import traceback
import re
import unicodedata

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain.docstore.document import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction

from evaluator.components.data_provider import QuerySpecification
from evaluator.config.schema import ModelConfig
from evaluator.utils.module_extractor import register_algorithm
from evaluator.interfaces.algorithm import Algorithm, AlgoResponse
from evaluator.utils.utils import log_verbose


# Constants

# Basic settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SEARCH_K = 10  # LLM dynamic search default
DEFAULT_MAX_RESULT_CHARS = 4000
DEFAULT_DROP_OLD_COLLECTION = True
DEFAULT_COLLECTION_NAME = "tool_fetcher_tools_collection"

# Similarity and indexing
DEFAULT_SIMILARITY_METRIC = "COSINE"  # COSINE, L2, or IP
DEFAULT_INDEX_TYPE = "FLAT"  # FLAT, HNSW, IVF_FLAT, etc.
DEFAULT_INDEXED_TOOL_DEF_PARTS = ["name", "description"]  # Can include: name, description, args, tags

# Hybrid search (dense + BM25 sparse)
DEFAULT_HYBRID_MODE = False
DEFAULT_ANALYZER_PARAMS = None  # BM25 analyzer parameters
DEFAULT_FUSION_TYPE = "rrf"  # "weighted" or "rrf"
DEFAULT_FUSION_K = 100  # k parameter for RRF fusion
DEFAULT_FUSION_ALPHA = 0.5  # alpha for weighted fusion (dense weight)

# Reranking
DEFAULT_CROSS_ENCODER_MODEL_NAME = None  # e.g., "BAAI/bge-reranker-large"
DEFAULT_RERANKER_POOL_SIZE = 50  # Retrieve this many before reranking to top_k

# Other limits and weights
MAX_EMBEDDING_TEXT_LENGTH = 2048  # Typical embedding model context limit
MIN_K = 1
MAX_K = 50
NAME_MATCH_WEIGHT = 2  # Substring search scoring weight for name matches
DESC_MATCH_WEIGHT = 1  # Substring search scoring weight for description matches

_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_WS_RE = re.compile(r"\s+")


class L2Wrapper:
    """Wrapper to L2-normalize embeddings for stable COSINE similarity."""
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


@register_algorithm("tool_fetcher")
class ToolFetcherAlgorithm(Algorithm):
    """
    Single-tool orchestration algorithm using dynamic tool retrieval.

    Exposes one tool (tool_hub) that the model can call to search and fetch
    other tools dynamically based on a natural-language request. The fetched
    tools are then made available to the agent on subsequent invocations of
    tool_hub during the same query.

    Retrieval policy: when multiple searches occur within a single query,
    the returned retrieved_tools list reflects the most recent search results.

    Optional configurable settings (to be provided in the 'settings' dictionary):

    - default_search_k: the number of tools to retrieve on a given search (default: 10).
    - embedding_model_id: the ID of the model to use for embedding calculation.
    - similarity_metric: the metric to use for embedding distance calculation - can be COSINE, L2 or IP (inner product).
    - index_type: the type of Milvus index to build - can be FLAT, HNSW, IVF_FLAT or any other supported type.
    - indexed_tool_def_parts: the parts of the tool definition to be used for index construction, such as 'name',
      'description', 'args', 'tags'.
    - text_preprocessing_operations: a list of preprocessing operations to apply on the indexed documents and the
      query text. The following operations are supported: 'unicode_normalization', 'lowercase', 'collapse_whitespaces',
      'split_camel_snake_case'. Set this parameter to None to disable preprocessing or to 'all' to enable all operations.
    - max_document_size: the maximal size, in characters, of a single indexed document, or None to disable the size limit.
    - hybrid_mode: True to enable hybrid (sparse + dense) search and False to only enable dense search.
    - analyzer_params: parameters for the Milvus BM25 analyzer.
    - fusion_type: the algorithm for combining the dense and the sparse scores if hybrid mode is activated. Milvus only
      supports "weighted" and "rrf".
    - fusion_alpha: the relative weight of the dense retriever score. The final score is calculated as
      alpha*dense + (1-alpha)*sparse. This parameter is only used with the "weighted" fusion type.
    - fusion_k: the k value to use for the "rrf" hybrid fusion mode.
    - cross_encoder_model_name: the name of the model to use for reranking or None to disable reranking.
    - reranker_pool_size: the number of results to retrieve from the vector DB before reranking. Must be greater than or
      equal to default_search_k.
    - tau: an optional threshold between 0 and 1 specifying the maximal allowed distance to the query. All tools having
      greater distance will be filtered. Set to None to disable this filter.
    - sim_threshold: an optional threshold between 0 and 1 specifying the maximal allowed similarity between the retrieved
      tools. Sets of highly similar tools will be filtered to only retain one. Set to None to disable this filter.
    - max_result_chars: maximum number of characters to return from tool invocation results (default: 4000).
    - drop_old_collection: whether to drop existing Milvus collection on startup (default: True).
    - collection_name: name of the Milvus collection to use (default: "tool_fetcher_tools_collection").
    """

    vector_store: Milvus or None
    reranker: CrossEncoderReranker or None
    embeddings: HuggingFaceEmbeddings or None
    tool_name_to_base_tool: Dict[str, BaseTool] or None

    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        super().__init__(settings, model_config, label)
        self.tool_name_to_base_tool = None
        self.vector_store = None
        self.embeddings = None
        self.reranker = None

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        super().set_up(model, tools)
        self.tool_name_to_base_tool = {t.name: t for t in tools}

        # Initialize reranker if configured
        if self._settings.get("cross_encoder_model_name"):
            log_verbose(f"Initializing reranker: {self._settings['cross_encoder_model_name']}")
            self.reranker = CrossEncoderReranker(
                model=HuggingFaceCrossEncoder(model_name=self._settings["cross_encoder_model_name"]),
                top_n=self._settings.get("default_search_k", DEFAULT_SEARCH_K),
            )

        self._build_vector_index(tools)

    def _build_vector_index(self, tools: List[BaseTool]) -> None:
        """Build Milvus vector index for tool retrieval. Falls back to None on failure."""
        try:
            embedding_model_id = self._settings.get("embedding_model_id", DEFAULT_EMBEDDING_MODEL)
            log_verbose(f"Initializing embeddings with model: {embedding_model_id}")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

            # Apply L2 normalization wrapper for COSINE similarity
            similarity_metric = self._settings.get("similarity_metric", "COSINE")
            if similarity_metric == "COSINE":
                embeddings = L2Wrapper(self.embeddings)
            else:
                embeddings = self.embeddings


            milvus_uri = os.getenv("MILVUS_URL") or "http://localhost:19530"
            collection = self._settings.get("collection_name", DEFAULT_COLLECTION_NAME)
            drop_old = bool(self._settings.get("drop_old_collection", DEFAULT_DROP_OLD_COLLECTION))

            # Build documents using configurable tool text composition
            docs = [
                Document(
                    page_content=self._compose_tool_text(t),
                    metadata={"name": t.name or ""}
                )
                for t in tools
            ]

            index_type = self._settings.get("index_type", "FLAT")
            index_params = {
                "index_type": index_type,
                "metric_type": similarity_metric,
            }
            search_params = {
                "metric_type": similarity_metric,
            }

            log_verbose(f"Building Milvus collection: {collection} (drop_old={drop_old})")

            if self._settings.get("hybrid_mode", False):
                # Hybrid search: dense + sparse (BM25)
                log_verbose("Enabling hybrid search (dense + BM25)")
                index_params = [index_params, None]
                search_params = [search_params, {}]

                analyzer_params = self._settings.get("analyzer_params")
                self.vector_store = Milvus.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    collection_name=collection,
                    connection_args={"uri": milvus_uri},
                    drop_old=drop_old,
                    index_params=index_params,
                    search_params=search_params,
                    builtin_function=BM25BuiltInFunction(analyzer_params=analyzer_params),
                    vector_field=["dense", "sparse"],
                )
            else:
                # Dense-only search
                self.vector_store = Milvus.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    collection_name=collection,
                    connection_args={"uri": milvus_uri},
                    drop_old=drop_old,
                    index_params=index_params,
                    search_params=search_params,
                )
        except Exception as e:
            log_verbose(f"Vector store initialization failed: {e}. Falling back to substring search.")
            log_verbose(traceback.format_exc())
            self.vector_store = None

    @staticmethod
    def _render_args_schema(schema: Dict[str, Any]) -> str:
        """Render tool arguments schema as concise string."""
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

    def _compose_tool_text(self, tool: BaseTool) -> str:
        """Compose indexed text from tool based on configured parts."""
        parts_to_include = self._settings.get("indexed_tool_def_parts", ["name", "description"])

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
            # Fallback to name + description
            segments = [f"name: {tool.name}", f"desc: {tool.description or ''}"]

        text = " | ".join(segments)
        # One-pass preprocess + truncation (configurable)
        text = self._preprocess_text(text)
        text = self._truncate(text)
        # Hard cap to embedding model max length for safety
        if len(text) > MAX_EMBEDDING_TEXT_LENGTH:
            text = text[:MAX_EMBEDDING_TEXT_LENGTH]
        return text

    def _preprocess_text(self, text: str) -> str:
        ops = self._settings.get("text_preprocessing_operations")
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
            t = t.replace('_', ' ')
            t = _CAMEL_RE.sub(' ', t)
        return t

    def _truncate(self, s: str, sep: str = " ... ") -> str:
        max_chars = self._settings.get("max_document_size")
        if max_chars is None or len(s) <= max_chars:
            return s
        half = (max_chars - len(sep)) // 2
        return s[:half] + sep + s[-half:]

    def _clamp_k(self, k: Optional[int], default: int) -> int:
        """Clamp k value to valid range."""
        try:
            value = int(k) if k is not None else default
        except (ValueError, TypeError):
            value = default
        return max(MIN_K, min(value, MAX_K))

    def _search_tools(self, query: str, limit: int) -> List[BaseTool]:
        """Search tools using vector similarity or substring matching."""
        if not self.tool_name_to_base_tool:
            return []

        # Try vector search first
        if self.vector_store is not None:
            try:
                # Determine pool size for reranking
                if self.reranker is not None:
                    pool_size = self._settings.get("reranker_pool_size", 50)
                    k = max(limit, pool_size)
                else:
                    k = limit

                # Perform vector search with optional hybrid fusion
                if self._settings.get("hybrid_mode", False):
                    fusion_type = self._settings.get("fusion_type", "rrf")
                    if fusion_type == "weighted":
                        alpha = self._settings.get("fusion_alpha", 0.5)
                        fusion_params = {"weights": [alpha, 1 - alpha]}
                    elif fusion_type == "rrf":
                        fusion_params = {"k": self._settings.get("fusion_k", 100)}
                    else:
                        raise ValueError(f"Unsupported hybrid fusion type: {fusion_type}")

                    results = self.vector_store.similarity_search_with_score(
                        self._preprocess_text(query or ""),
                        k=k,
                        ranker_type=fusion_type,
                        ranker_params=fusion_params,
                    )
                else:
                    results = self.vector_store.similarity_search_with_score(
                        self._preprocess_text(query or ""), k=k
                    )

                # Keep only known tools and collect scores
                docs_and_scores = [(doc, score) for (doc, score) in results if doc.metadata.get("name") in self.tool_name_to_base_tool]

                # Apply tau thresholding if configured
                tau = self._settings.get("tau")
                metric = self._settings.get("similarity_metric", "COSINE")
                if tau is not None:
                    if str(metric).upper() in {"COSINE", "IP"}:
                        docs_and_scores = [(d, s) for (d, s) in docs_and_scores if s >= float(tau)]
                        docs_and_scores.sort(key=lambda x: x[1], reverse=True)
                    elif str(metric).upper() == "L2":
                        docs_and_scores = [(d, s) for (d, s) in docs_and_scores if s <= float(tau)]
                        docs_and_scores.sort(key=lambda x: x[1])
                    else:
                        raise ValueError(f"Unknown metric: {metric}")

                # Strip to docs before reranking
                docs = [d for (d, _s) in docs_and_scores]

                # Apply reranking if enabled
                if self.reranker is not None and docs:
                    docs = self.reranker.compress_documents(docs, self._preprocess_text(query or ""))

                # Apply NMS-based diversity filtering if configured
                sim_threshold = self._settings.get("sim_threshold")
                if sim_threshold is not None and docs:
                    docs = self._diversify_nms(docs)

                # Convert back to tools
                ordered = [self.tool_name_to_base_tool[doc.metadata["name"]] for doc in docs[:limit]]
                if ordered:
                    return ordered
            except Exception as e:
                log_verbose(f"Vector search failed: {e}. Falling back to substring search.")
                log_verbose(traceback.format_exc())

        # Fallback: substring search
        all_tools = list(self.tool_name_to_base_tool.values())
        q = (query or "").strip().lower()
        ranked = []
        for tool in all_tools:
            name = (tool.name or "").lower()
            desc = (getattr(tool, "description", "") or "").lower()
            score = (NAME_MATCH_WEIGHT if q in name else 0) + (DESC_MATCH_WEIGHT if q in desc else 0)
            if score > 0:
                ranked.append((score, tool))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked[:limit]] or all_tools[:limit]

    def _embed_docs(self, docs: List[Document]) -> np.ndarray:
        """Encode docs -> (n, d), then L2-normalize for cosine similarity."""
        texts = [d.page_content for d in docs]
        vecs = np.asarray(self.embeddings.embed_documents(texts), dtype=np.float32)
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(n, 1e-12)

    def _diversify_nms(self, docs: List[Document]) -> List[Document]:
        """Apply NMS-based diversity filtering to remove highly similar tools."""
        if not docs:
            return []
        embedded = self._embed_docs(docs)
        kept_indices = []
        threshold = float(self._settings.get("sim_threshold"))
        for i in range(len(docs)):
            if not kept_indices:
                kept_indices.append(i)
                continue
            v = embedded[i]
            sims = (embedded[kept_indices] @ v)
            if float(np.max(sims)) < threshold:
                kept_indices.append(i)
        return [docs[i] for i in kept_indices]

    def _handle_search(self, query: str, k: Optional[int], active_tools: List[BaseTool], retrieved_tools: List[str]) -> str:
        """Handle tool search action."""
        default_k = self._settings.get("default_search_k", DEFAULT_SEARCH_K)
        limit = self._clamp_k(k, default_k)
        matches = self._search_tools(query, limit)

        existing_names = {t.name for t in active_tools}
        newly_added = []
        for t in matches:
            if t.name not in existing_names:
                active_tools.append(t)
                newly_added.append(t.name)

        # Update retrieval-only ranked list for metrics (most recent search policy)
        retrieved_tools[:] = [t.name for t in matches]

        return json.dumps({
            "mode": "search",
            "fetched": newly_added,
            "active": [t.name for t in active_tools],
        })

    def _handle_call(self, tool_name: str, tool_input: str, active_tools: List[BaseTool], retrieved_tools: List[str]) -> str:
        """Handle tool invocation action."""
        tool = (self.tool_name_to_base_tool or {}).get(tool_name)

        if tool is None:
            return json.dumps({"mode": "call", "error": f"tool '{tool_name}' not found"})

        # Parse input as JSON if possible
        try:
            parsed = json.loads(tool_input) if tool_input else tool_input
        except json.JSONDecodeError:
            parsed = tool_input

        # Add to active tools
        if tool.name not in {t.name for t in active_tools}:
            active_tools.append(tool)

        # Log tool usage
        self._log_tool_usage(tool.name)

        # Invoke tool
        try:
            result = tool.invoke(parsed)
            result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            log_verbose(f"Tool invocation failed for {tool.name}: {e}")
            log_verbose(traceback.format_exc())
            return json.dumps({"mode": "call", "tool": tool.name, "error": str(e)})

        max_chars = self._settings.get("max_result_chars", DEFAULT_MAX_RESULT_CHARS)
        truncated = len(result_str) > max_chars
        return json.dumps({
            "mode": "call",
            "tool": tool.name,
            "result": result_str[:max_chars],
            "truncated": truncated,
        })

    def _log_tool_usage(self, tool_name: str) -> None:
        """Log tool usage to file if TOOL_LOG_PATH is set."""
        try:
            log_path = os.getenv("TOOL_LOG_PATH")
            if log_path:
                with open(log_path, "a") as f:
                    f.write(f"[TOOL] {tool_name}\n")
        except Exception as e:
            log_verbose(f"Tool logging failed: {e}")

    def _make_tool_hub(self, active_tools: List[BaseTool], retrieved_tools: List[str]) -> BaseTool:
        """
        Create the single tool_hub tool for searching and calling other tools.

        The returned closure captures self for accessing instance state (_all_tools,
        _settings, etc.) and the query-scoped active_tools list.
        """
        def run(action: str = "", query: str = "", k: Optional[int] = None,
                tool_name: str = "", tool_input: str = "") -> str:
            act = (action or "").strip().lower()

            if act in ("search", "find", "fetch") or (not act and query):
                return self._handle_search(query, k, active_tools, retrieved_tools)

            if act == "call" or tool_name:
                return self._handle_call(tool_name, tool_input, active_tools, retrieved_tools)

            return json.dumps({"error": "invalid action; use 'search' or 'call'"})

        return StructuredTool.from_function(
            name="tool_hub",
            description=(
                "IMPORTANT: This is the ONLY tool you can call directly. All other tools must be accessed through tool_hub.\n\n"
                "To complete any task:\n"
                "1. FIRST search for relevant tools: action='search', query='description of what you need', k=10\n"
                "2. THEN call the found tools: action='call', tool_name='exact_tool_name', tool_input='{\"param\": \"value\"}'\n\n"
                "The search will return a list of available tools. You must then call each tool using action='call'."
            ),
            func=run,
        )

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        """Process query using the tool hub pattern."""
        if self.tool_name_to_base_tool is None:
            raise RuntimeError("process_query called before set_up")

        # Create query-scoped active tools and retrieval-only lists (fixes concurrency and keeps metrics clean)
        active_tools = []
        retrieved_tools: List[str] = []

        # Create agent with tool_hub
        hub = self._make_tool_hub(active_tools, retrieved_tools)
        agent = create_react_agent(self._model, [hub])

        # No additional guidance - rely solely on tool_hub's built-in description
        # to isolate retrieval quality from prompt engineering effects
        response = await self._invoke_agent_on_query(agent, query_spec.query)

        # Return the retrieval-only ranked list for metrics (most recent search)
        return response, list(retrieved_tools)

    def tear_down(self) -> None:
        """Clean up resources."""
        self.tool_name_to_base_tool = None
        self.vector_store = None
        self.embeddings = None
        self.reranker = None

    def get_default_settings(self) -> Dict[str, Any]:
        """Return default configuration settings."""
        return {
            # Basic settings
            "embedding_model_id": DEFAULT_EMBEDDING_MODEL,
            "default_search_k": DEFAULT_SEARCH_K,
            "drop_old_collection": DEFAULT_DROP_OLD_COLLECTION,
            "collection_name": DEFAULT_COLLECTION_NAME,
            "max_result_chars": DEFAULT_MAX_RESULT_CHARS,

            # Similarity and indexing
            "similarity_metric": DEFAULT_SIMILARITY_METRIC,
            "index_type": DEFAULT_INDEX_TYPE,
            "indexed_tool_def_parts": DEFAULT_INDEXED_TOOL_DEF_PARTS,

            # Preprocessing
            "text_preprocessing_operations": None,
            "max_document_size": None,

            # Hybrid search (dense + BM25 sparse)
            "hybrid_mode": DEFAULT_HYBRID_MODE,
            "analyzer_params": DEFAULT_ANALYZER_PARAMS,
            "fusion_type": DEFAULT_FUSION_TYPE,
            "fusion_k": DEFAULT_FUSION_K,
            "fusion_alpha": DEFAULT_FUSION_ALPHA,

            # Reranking
            "cross_encoder_model_name": DEFAULT_CROSS_ENCODER_MODEL_NAME,
            "reranker_pool_size": DEFAULT_RERANKER_POOL_SIZE,

            # Post-retrieval filtering
            "tau": None,
            "sim_threshold": None,
        }


