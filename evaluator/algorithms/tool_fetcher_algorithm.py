import json
import os
import traceback
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from evaluator.components.data_provider import QuerySpecification
from evaluator.config.schema import ModelConfig
from evaluator.utils.module_extractor import register_algorithm
from evaluator.interfaces.algorithm import AlgoResponse
from evaluator.utils.utils import log_verbose
from evaluator.algorithms.base_retrieval_algorithm import BaseRetrievalAlgorithm


# Constants
DEFAULT_SEARCH_K = 10  # LLM dynamic search default
DEFAULT_MAX_RESULT_CHARS = 4000
DEFAULT_DROP_OLD_COLLECTION = True
DEFAULT_COLLECTION_NAME = "tool_fetcher_tools_collection"
DEFAULT_RECURSION_LIMIT = 10  # Tool Fetcher needs more iterations due to search-then-call pattern

# Substring search fallback weights
MIN_K = 1
MAX_K = 50
NAME_MATCH_WEIGHT = 2
DESC_MATCH_WEIGHT = 1

# Tool summary settings
SUMMARY_MAX_DESCRIPTION = 100  # Increased to include parameter info in tool descriptions

# Agent system prompt
AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that uses tools to answer queries efficiently.\n\n"
    "REASONING APPROACH:\n"
    "Before each action, think through:\n"
    "• What information do I need to answer this query?\n"
    "• Which tool would best provide that information?\n"
    "• Do I have the required parameters, or should I ask the user?\n\n"
    "WORKFLOW:\n"
    "1. SEARCH: Find relevant tools using action='search' with a specific query\n"
    "2. ANALYZE: Review the returned tools and select the most appropriate one\n"
    "3. CALL: Execute the tool with proper parameters using action='call'\n"
    "4. RESPOND: Provide a clear answer based on the tool results\n\n"
    "GUIDELINES:\n"
    "• Search first with specific keywords (e.g., 'weather forecast' not 'weather')\n"
    "• If a tool call fails or returns empty results, explain what happened\n"
    "• If missing required parameters, ask the user rather than guessing\n"
    "• For complex tasks, you may search multiple times with different queries\n"
    "• Aim for efficiency, but prioritize correctness over speed\n"
)

TOOL_HUB_DESCRIPTION = (
    "Universal tool hub for dynamic tool discovery and execution.\n\n"
    "IMPORTANT: This is your ONLY available tool. You MUST search before calling any tools.\n\n"
    "STRATEGY: Before each action, think: What capability do I need? Which tools might help? "
    "What parameters are required?\n\n"
    "═══ WORKFLOW ═══\n"
    "1️⃣ SEARCH: Find tools matching your task\n"
    "   Parameters:\n"
    "   • action='search'\n"
    "   • query='<specific task description>'\n"
    "     → Be specific: 'get current weather forecast' not just 'weather'\n"
    "     → Use keywords: 'calculate mortgage payment' not 'help with mortgage'\n"
    "   • k=<number> (optional, default 10) – how many tools to retrieve\n"
    "     → Refine keywords or adjust k if results look off\n\n"
    "   Returns: {\"mode\": \"search\", \"status\": \"success\", \"payload\": {\"tools\": [...]}}\n\n"
    "2️⃣ CALL: Execute a tool from your search results\n"
    "   Parameters:\n"
    "   • action='call'\n"
    "   • tool_name='<exact name from search results>'\n"
    "   • tool_input='<JSON object with parameters>'\n"
    "     → Must be valid JSON: {\"key\": \"value\"}\n"
    "     → Include all required fields from tool description\n\n"
    "   Returns: {\"mode\": \"call\", \"status\": \"success\", \"payload\": {\"result\": \"...\"}}\n\n"
    "═══ EXAMPLE ═══\n"
    "Task: 'Get weather for Seattle'\n"
    "1. Search: action='search', query='get weather forecast for city', k=10\n"
    "2. Review: Check tools array → pick 'get_weather_by_city' (closest match)\n"
    "3. Call: action='call', tool_name='get_weather_by_city', tool_input='{\"city\": \"Seattle\"}'\n\n"
    "═══ BEST PRACTICES ═══\n"
    "• Read tool descriptions carefully—choose the most specific match\n"
    "• Before calling: verify tool description matches your exact need\n"
    "• Run additional searches with different keywords when needed\n"
    "• Iterate search → call → search for multi-step tasks\n"
    "• On 'error' status:\n"
    "  - Missing tool? Search with broader/different keywords\n"
    "  - Invalid input? Check tool description for correct parameter format\n"
    "  - Wrong result? Review descriptions and try a more specific tool"
)


@register_algorithm("tool_fetcher")
class ToolFetcherAlgorithm(BaseRetrievalAlgorithm):
    """
    Single-tool orchestration algorithm using dynamic tool retrieval.

    Exposes one tool (tool_hub) that enables search and execution of other tools
    based on natural-language queries. Retrieved tools accumulate across multiple
    searches within a single query.

    Algorithm-specific settings:
    - default_search_k: Number of tools to retrieve per search (default: 10)
    - max_result_chars: Character limit for tool results (default: 4000)
    - collection_name: Milvus collection name (default: "tool_fetcher_tools_collection")
    - drop_old_collection: Drop existing collection on startup (default: True)
    - recursion_limit: Maximum iterations for LangGraph agent (default: 10)

    Inherits retrieval settings from BaseRetrievalAlgorithm:
    - Embedding configuration: embedding_model_id, similarity_metric
    - Indexing: index_type, indexed_tool_def_parts, text_preprocessing_operations, max_document_size
    - Hybrid search: hybrid_mode, analyzer_params, fusion_type, fusion_alpha, fusion_k
    - Reranking: cross_encoder_model_name, reranker_pool_size
    - Filtering: tau (distance threshold), sim_threshold (similarity deduplication)
    """

    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        super().__init__(settings, model_config, label)

    def get_default_settings(self) -> Dict[str, Any]:
        """Extend base settings with ToolFetcher-specific settings."""
        base_settings = super().get_default_settings()
        base_settings.update({
            # ToolFetcher specific
            "default_search_k": DEFAULT_SEARCH_K,
            "drop_old_collection": DEFAULT_DROP_OLD_COLLECTION,
            "collection_name": DEFAULT_COLLECTION_NAME,
            "max_result_chars": DEFAULT_MAX_RESULT_CHARS,
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
        })
        return base_settings

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        super().set_up(model, tools)
        self._update_tool_mapping(tools)

        # Initialize reranker
        default_k = self._settings.get("default_search_k", DEFAULT_SEARCH_K)
        self._initialize_reranker(default_k)

        # Build Milvus vector index with fallback to substring search
        try:
            collection = self._settings.get("collection_name", DEFAULT_COLLECTION_NAME)
            drop_old = bool(self._settings.get("drop_old_collection", DEFAULT_DROP_OLD_COLLECTION))
            self._build_vector_store(tools, collection, drop_old)
        except Exception as e:
            log_verbose(f"Vector store initialization failed: {e}. Falling back to substring search.")
            log_verbose(traceback.format_exc())
            self.vector_store = None

    def _search_tools(self, query: str, limit: int) -> List[BaseTool]:
        """Search tools using vector similarity or substring matching."""
        if not self.tool_name_to_base_tool:
            return []

        raw_query = (query or "").strip()
        normalized_query = self._preprocess_text(raw_query)

        # Try vector search first
        if self.vector_store is not None:
            try:
                # Determine pool size for reranking using base class helper
                k = self._get_retrieval_k(limit)

                docs_and_scores = self._vector_search_with_scores(normalized_query, k)

                if docs_and_scores:
                    # Use base class postprocessing (handles tau thresholding, reranking, and NMS)
                    docs = self._postprocess_results(
                        docs_and_scores,
                        normalized_query,
                        limit,
                    )

                    # Convert back to tools using base class helper
                    ordered = self._docs_to_tools(docs, limit)

                    if ordered:
                        return ordered
            except Exception as e:
                log_verbose(f"Vector search failed: {e}. Falling back to substring search.")
                log_verbose(traceback.format_exc())

        # Fallback: substring search using base class helper
        return self._substring_search_tools(raw_query, limit, NAME_MATCH_WEIGHT, DESC_MATCH_WEIGHT)

    def _handle_search(self, query: str, k: Optional[int], active_tools: List[BaseTool],
                       retrieved_tools: List[str]) -> str:
        """Handle tool search action."""
        default_k = self._settings.get("default_search_k", DEFAULT_SEARCH_K)
        # Clamp k to valid range [MIN_K, MAX_K]
        try:
            limit = int(k) if k is not None else default_k
        except (ValueError, TypeError):
            limit = default_k
        limit = max(MIN_K, min(limit, MAX_K))

        matches = self._search_tools(query, limit)

        existing_names = {t.name for t in active_tools}
        newly_added = []

        retrieved_seen = set(retrieved_tools)
        tool_summaries = []
        for t in matches:
            desc = (getattr(t, "description", "") or "No description available").strip()
            if len(desc) > SUMMARY_MAX_DESCRIPTION:
                desc = f"{desc[:SUMMARY_MAX_DESCRIPTION]}..."
            tool_summaries.append({"name": t.name, "description": desc})

            if t.name not in existing_names:
                active_tools.append(t)
                newly_added.append(t.name)
                existing_names.add(t.name)

            if t.name not in retrieved_seen:
                retrieved_tools.append(t.name)
                retrieved_seen.add(t.name)
        
        log_verbose(f"Search query: '{query}' | Retrieved {len(matches)} tools | "
                    f"Total unique retrieved: {len(retrieved_tools)} | Newly added: {len(newly_added)}")

        return json.dumps({
            "mode": "search",
            "status": "success",
            "payload": {
                "message": f"Found {len(matches)} tools matching '{query}'",
                "tools": tool_summaries,
                "newly_added_count": len(newly_added),
                "total_active_count": len(active_tools),
            }
        })

    async def _handle_call(self, tool_name: str, tool_input: str, active_tools: List[BaseTool]) -> str:
        """Handle tool invocation action with proper parameter validation."""
        active_names = {t.name for t in active_tools}
        # Enforce search-before-call to ensure fair retrieval metrics
        if tool_name not in active_names:
            return json.dumps({
                "mode": "call",
                "status": "error",
                "payload": {
                    "error": f"Tool '{tool_name}' not in your active list.",
                    "hint": "You must search for tools before calling them. Use action='search' first.",
                    "active_tools_count": len(active_tools),
                }
            })

        tool = self.tool_name_to_base_tool.get(tool_name)
        if tool is None:
            available = list(self.tool_name_to_base_tool.keys()) if self.tool_name_to_base_tool else []
            return json.dumps({
                "mode": "call",
                "status": "error",
                "payload": {
                    "error": f"Tool '{tool_name}' not found in registry.",
                    "hint": f"Available tools count: {len(available)}. Try searching again.",
                }
            })

        # Parse tool input with robust JSON extraction
        try:
            if isinstance(tool_input, str):
                tool_input = tool_input.strip()
                if not tool_input or tool_input == "{}":
                    parsed = {}
                else:
                    # Use safe parser that can extract JSON from surrounding text
                    parsed = self._safe_json_parse(tool_input)
                    if parsed is None or not isinstance(parsed, dict):
                        return json.dumps({
                            "mode": "call",
                            "status": "error",
                            "payload": {
                                "tool": tool.name,
                                "error": f"Invalid JSON in tool_input. Could not parse: {tool_input[:100]}",
                                "hint": "Ensure tool_input is valid JSON like {\"param\": \"value\"}",
                            }
                        })
            elif isinstance(tool_input, dict):
                parsed = tool_input
            else:
                parsed = {}
        except Exception as e:
            return json.dumps({
                "mode": "call",
                "status": "error",
                "payload": {
                    "tool": tool.name,
                    "error": f"Failed to parse tool_input: {str(e)}",
                    "hint": "tool_input must be a JSON object",
                }
            })

        # Check for required parameters using the tool's schema (JSON Schema format)
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            schema_dict = args_schema if isinstance(args_schema, dict) else None
            if schema_dict is None and hasattr(args_schema, "schema"):
                try:
                    schema_dict = args_schema.schema()
                except Exception:
                    schema_dict = None

            if isinstance(schema_dict, dict):
                required_fields = set(schema_dict.get("required", []))
                missing = required_fields - set(parsed.keys())
                if missing:
                    return json.dumps({
                        "mode": "call",
                        "status": "error",
                        "payload": {
                            "tool": tool.name,
                            "error": f"Missing required parameters: {', '.join(sorted(missing))}",
                            "required": sorted(required_fields),
                            "hint": "Check the tool description or ask the user for these values.",
                        }
                    })

        # Log tool usage for metrics
        try:
            log_path = os.getenv("TOOL_LOG_PATH")
            if log_path:
                with open(log_path, "a") as f:
                    f.write(f"[TOOL] {tool.name}\n")
        except Exception as e:
            log_verbose(f"Tool logging failed: {e}")

        # Normalize 'tool_input' field to JSON string when present to avoid adapter issues
        try:
            if isinstance(parsed, dict) and "tool_input" in parsed:
                ti_val = parsed["tool_input"]
                if isinstance(ti_val, dict):
                    parsed["tool_input"] = json.dumps(ti_val)
                elif isinstance(ti_val, str):
                    # If it's a string that looks like a Python dict, try to parse then re-dump as JSON
                    maybe = self._safe_json_parse(ti_val)
                    if isinstance(maybe, dict):
                        parsed["tool_input"] = json.dumps(maybe)
        except Exception:
            # best-effort normalization only; ignore errors
            pass

        log_verbose(f"Calling tool '{tool.name}' with input: {str(parsed)[:100]}...")

        # Invoke tool asynchronously
        try:
            result = await tool.ainvoke(parsed)
            result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            error_msg = str(e)
            log_verbose(f"Tool invocation failed for {tool.name}: {error_msg}")
            log_verbose(traceback.format_exc())
            return json.dumps({
                "mode": "call",
                "status": "error",
                "payload": {
                    "tool": tool.name,
                    "error": error_msg,
                    "hint": "The tool execution failed. Verify your parameters match the tool's requirements.",
                }
            })

        # Check for empty or suspicious results
        result_analysis = self._analyze_tool_result(result_str, tool.name)
        if not result_analysis["is_valid"]:
            log_verbose(f"Tool '{tool.name}' returned suspicious result: {result_analysis['issue']}")
            return json.dumps({
                "mode": "call",
                "status": "warning",
                "payload": {
                    "tool": tool.name,
                    "issue": result_analysis["issue"],
                    "result": result_str[:500],
                    "suggestion": result_analysis["suggestion"],
                }
            })

        max_chars = self._settings.get("max_result_chars", DEFAULT_MAX_RESULT_CHARS)
        truncated = len(result_str) > max_chars

        log_verbose(f"Tool '{tool.name}' executed successfully. Result length: {len(result_str)} chars")

        return json.dumps({
            "mode": "call",
            "status": "success",
            "payload": {
                "tool": tool.name,
                "result": result_str[:max_chars],
                "truncated": truncated,
            }
        })

    @staticmethod
    def _analyze_tool_result(result_str: str, tool_name: str) -> Dict[str, Any]:
        """Detect empty, truncated, or malformed API responses."""
        result_str = (result_str or "").strip()
        
        # Empty result
        if not result_str or result_str in ('{}', '[]', '""', "''", 'null', 'None'):
            return {
                "is_valid": False,
                "issue": "Empty result",
                "suggestion": f"The {tool_name} tool returned no data. This could mean: (1) No results match your query, (2) The service is unavailable, or (3) Try broader search parameters."
            }
        
        # Truncated JSON (starts with quote instead of { or [)
        if result_str.startswith(('"', "'")):
            return {
                "is_valid": False,
                "issue": "Truncated or malformed response",
                "suggestion": "API returned incomplete data. Try again or use an alternative tool."
            }
        
        # Suspiciously short
        if len(result_str) < 10 and not result_str.startswith(('{', '[')):
            return {
                "is_valid": False,
                "issue": "Incomplete response",
                "suggestion": f"Result '{result_str[:50]}' seems incomplete. Verify parameters."
            }
        
        return {"is_valid": True, "issue": None, "suggestion": None}

    def _make_tool_hub(self, active_tools: List[BaseTool], retrieved_tools: List[str]) -> BaseTool:
        """
        Create the single tool_hub tool for searching and calling other tools.

        The returned closure captures self for accessing instance state and the 
        query-scoped active_tools list.
        """
        async def run(action: str = "", query: str = "", k: Optional[int] = None,
                      tool_name: str = "", tool_input: str = "") -> str:
            act = (action or "").strip().lower()

            if act in ("search", "find", "fetch") or (not act and query):
                return self._handle_search(query, k, active_tools, retrieved_tools)

            if act == "call" or tool_name:
                return await self._handle_call(tool_name, tool_input, active_tools)

            return json.dumps({"error": "invalid action; use 'search' or 'call'"})

        return StructuredTool.from_function(
            name="tool_hub",
            description=TOOL_HUB_DESCRIPTION,
            coroutine=run,
        )

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        """Process query using a tool_hub-backed ReAct agent."""
        if self.tool_name_to_base_tool is None:
            raise RuntimeError("process_query called before set_up")

        # Query-scoped state (important for concurrency and clean metrics)
        active_tools: List[BaseTool] = []
        retrieved_tools: List[str] = []

        hub = self._make_tool_hub(active_tools, retrieved_tools)

        agent = create_react_agent(
            self._model,
            [hub],
            prompt=AGENT_SYSTEM_PROMPT
        )
        result = await self._invoke_agent_on_query(
            agent,
            query_spec.query,
            config={
                "recursion_limit": self._settings.get("recursion_limit", DEFAULT_RECURSION_LIMIT),
            }
        )

        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            final_msg = messages[-1]
            final_preview = getattr(final_msg, "content", "")
            log_verbose(f"Final response: {str(final_preview)[:500]}...")
        else:
            log_verbose(f"Final response: {str(result)[:500]}...")

        # Ensure retrieved tools list has no duplicates while preserving order
        return result, self._dedup_keep_order(retrieved_tools)

    def tear_down(self) -> None:
        """Clean up resources."""
        self.tool_name_to_base_tool = None
        self.vector_store = None
        self.embeddings = None
        self.reranker = None