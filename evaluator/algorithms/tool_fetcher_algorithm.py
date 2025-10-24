from typing import Dict, List, Any, Optional
import json
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from evaluator.components.data_provider import QuerySpecification
from evaluator.config.schema import ModelConfig
from evaluator.utils.module_extractor import register_algorithm
from evaluator.interfaces.algorithm import Algorithm, AlgoResponse
from evaluator.utils.utils import log_verbose


# Constants
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SEARCH_K = 8  # LLM dynamic search default
DEFAULT_MAX_RESULT_CHARS = 4000
DEFAULT_DROP_OLD_COLLECTION = True
DEFAULT_COLLECTION_NAME = "tool_fetcher_tools_collection"
MAX_EMBEDDING_TEXT_LENGTH = 2048  # Typical embedding model context limit
MIN_K = 1
MAX_K = 50
# Substring search scoring weights
NAME_MATCH_WEIGHT = 2
DESC_MATCH_WEIGHT = 1


@register_algorithm("tool_fetcher")
class ToolFetcherAlgorithm(Algorithm):
    """
    Single-tool orchestration algorithm.

    Exposes one tool (tool_hub) that the model can call to search and fetch
    other tools dynamically based on a natural-language request. The fetched
    tools are then made available to the agent on subsequent invocations of
    tool_hub during the same query.
    """

    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        super().__init__(settings, model_config, label)
        self._all_tools = None
        self._tool_map = None  # Cache for name->tool lookup
        self._active_tools = None
        self._vector_store = None
        self._embeddings = None

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        super().set_up(model, tools)
        self._all_tools = tools
        self._tool_map = {t.name: t for t in tools}  # Build lookup cache once
        self._active_tools = []
        self._build_vector_index(tools)

    def _build_vector_index(self, tools: List[BaseTool]) -> None:
        """Build Milvus vector index for tool retrieval. Falls back to None on failure."""
        try:
            embedding_model_id = self._settings.get("embedding_model_id", DEFAULT_EMBEDDING_MODEL)
            log_verbose(f"Initializing embeddings with model: {embedding_model_id}")
            self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

            milvus_uri = os.getenv("MILVUS_URL") or "http://localhost:19530"
            collection = self._settings.get("collection_name", DEFAULT_COLLECTION_NAME)
            drop_old = bool(self._settings.get("drop_old_collection", DEFAULT_DROP_OLD_COLLECTION))

            docs = [
                Document(
                    page_content=f"name: {t.name or ''} | desc: {getattr(t, 'description', '') or ''}"[:MAX_EMBEDDING_TEXT_LENGTH],
                    metadata={"name": t.name or ""}
                )
                for t in tools
            ]

            log_verbose(f"Building Milvus collection: {collection} (drop_old={drop_old})")
            self._vector_store = Milvus.from_documents(
                documents=docs,
                embedding=self._embeddings,
                collection_name=collection,
                connection_args={"uri": milvus_uri},
                drop_old=drop_old,
                index_params={"index_type": "FLAT", "metric_type": "COSINE"},
                search_params={"metric_type": "COSINE"},
            )
        except Exception as e:
            log_verbose(f"Vector store initialization failed: {e}. Falling back to substring search.")
            self._vector_store = None

    def _clamp_k(self, k: Optional[int], default: int) -> int:
        """Clamp k value to valid range."""
        try:
            value = int(k) if k is not None else default
        except (ValueError, TypeError):
            value = default
        return max(MIN_K, min(value, MAX_K))

    def _search_tools(self, query: str, limit: int) -> List[BaseTool]:
        """Search tools using vector similarity or substring matching."""
        if not self._all_tools:
            return []

        # Try vector search first
        if self._vector_store is not None:
            try:
                results = self._vector_store.similarity_search_with_score(query or "", k=limit)
                ordered = [
                    self._tool_map[doc.metadata["name"]]
                    for doc, _score in results
                    if doc.metadata.get("name") in self._tool_map
                ]
                if ordered:
                    return ordered
            except Exception as e:
                log_verbose(f"Vector search failed: {e}. Falling back to substring search.")

        # Fallback: substring search
        q = (query or "").strip().lower()
        ranked = []
        for tool in self._all_tools:
            name = (tool.name or "").lower()
            desc = (getattr(tool, "description", "") or "").lower()
            score = (NAME_MATCH_WEIGHT if q in name else 0) + (DESC_MATCH_WEIGHT if q in desc else 0)
            if score > 0:
                ranked.append((score, tool))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked[:limit]] or self._all_tools[:limit]

    def _handle_search(self, query: str, k: Optional[int]) -> str:
        """Handle tool search action."""
        default_k = self._settings.get("default_search_k", DEFAULT_SEARCH_K)
        limit = self._clamp_k(k, default_k)
        matches = self._search_tools(query, limit)

        existing_names = {t.name for t in self._active_tools}
        newly_added = []
        for t in matches:
            if t.name not in existing_names:
                self._active_tools.append(t)
                newly_added.append(t.name)

        return json.dumps({
            "mode": "search",
            "fetched": newly_added,
            "active": [t.name for t in self._active_tools],
        })

    def _handle_call(self, tool_name: str, tool_input: str) -> str:
        """Handle tool invocation action."""
        tool = self._tool_map.get(tool_name)

        if tool is None:
            return json.dumps({"mode": "call", "error": f"tool '{tool_name}' not found"})

        # Parse input as JSON if possible
        try:
            parsed = json.loads(tool_input) if tool_input else tool_input
        except json.JSONDecodeError:
            parsed = tool_input

        # Add to active tools
        if tool.name not in {t.name for t in self._active_tools}:
            self._active_tools.append(tool)

        # Log tool usage
        self._log_tool_usage(tool.name)

        # Invoke tool
        try:
            result = tool.invoke(parsed)
            result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            log_verbose(f"Tool invocation failed for {tool.name}: {e}")
            return json.dumps({"mode": "call", "tool": tool.name, "error": str(e)})

        max_chars = self._settings.get("max_result_chars", DEFAULT_MAX_RESULT_CHARS)
        return json.dumps({
            "mode": "call",
            "tool": tool.name,
            "result": result_str[:max_chars],
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

    def _make_tool_hub(self) -> BaseTool:
        """
        Create the single tool_hub tool for searching and calling other tools.

        The returned closure captures self for accessing instance state (_all_tools,
        _active_tools, _settings, etc.) and delegates to _handle_search/_handle_call.
        """
        def run(action: str = "", query: str = "", k: Optional[int] = None,
                tool_name: str = "", tool_input: str = "") -> str:
            act = (action or "").strip().lower()

            if act in ("search", "find", "fetch") or (not act and query):
                return self._handle_search(query, k)

            if act == "call" or tool_name:
                return self._handle_call(tool_name, tool_input)

            return json.dumps({"error": "invalid action; use 'search' or 'call'"})

        return StructuredTool.from_function(
            name="tool_hub",
            description=(
                "IMPORTANT: This is the ONLY tool you can call directly. All other tools must be accessed through tool_hub.\n\n"
                "To complete any task:\n"
                "1. FIRST search for relevant tools: action='search', query='description of what you need', k=8\n"
                "2. THEN call the found tools: action='call', tool_name='exact_tool_name', tool_input='{\"param\": \"value\"}'\n\n"
                "The search will return a list of available tools. You must then call each tool using action='call'."
            ),
            func=run,
        )

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        """Process query using the tool hub pattern."""
        if self._all_tools is None:
            raise RuntimeError("process_query called before set_up")

        # Reset active tools
        self._active_tools = []

        # Create agent with tool_hub
        hub = self._make_tool_hub()
        agent = create_react_agent(self._model, [hub])

        # No additional guidance - rely solely on tool_hub's built-in description
        # to isolate retrieval quality from prompt engineering effects
        response = await self._invoke_agent_on_query(agent, query_spec.query)

        # Return tools that the agent actually retrieved during execution
        retrieved = [t.name for t in self._active_tools]
        return response, retrieved

    def tear_down(self) -> None:
        """Clean up resources."""
        self._all_tools = None
        self._tool_map = None
        self._active_tools = None
        self._vector_store = None
        self._embeddings = None

    def get_default_settings(self) -> Dict[str, Any]:
        """Return default configuration settings."""
        return {
            "embedding_model_id": DEFAULT_EMBEDDING_MODEL,
            "default_search_k": DEFAULT_SEARCH_K,
            "drop_old_collection": DEFAULT_DROP_OLD_COLLECTION,
            "collection_name": DEFAULT_COLLECTION_NAME,
            "max_result_chars": DEFAULT_MAX_RESULT_CHARS,
        }


