import asyncio
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
DEFAULT_SEARCH_K = 10  # Default number of tools to retrieve per search
DEFAULT_MAX_RESULT_CHARS = 4000
DEFAULT_DROP_OLD_COLLECTION = True
DEFAULT_COLLECTION_NAME = "tool_fetcher_tools_collection"
DEFAULT_RECURSION_LIMIT = 20  # Tool Fetcher needs more iterations due to search-then-call pattern.

# Substring search fallback weights
MIN_K = 1
MAX_K = 50
NAME_MATCH_WEIGHT = 2
DESC_MATCH_WEIGHT = 1

# Tool summary settings
SUMMARY_MAX_DESCRIPTION = 100  # Truncation limit for tool descriptions in search results

# Search loop prevention
MAX_CONSECUTIVE_SEARCHES = 2  # Warn agent after this many searches without a tool call

# Agent system prompt
AGENT_SYSTEM_PROMPT = (
    "You are a tool-using assistant. For EVERY query, you MUST execute tools - explanations alone are INSUFFICIENT.\n\n"
    "═══ MANDATORY WORKFLOW ═══\n"
    "1. SEARCH: Find relevant tools using action='search'\n"
    "2. EXECUTE: Immediately call the relevant tools using action='call'\n"
    "3. RESPOND: Answer based on tool results\n\n"
    "🚨 CRITICAL: You CANNOT answer queries without executing tools. Explaining what tools to use is NOT acceptable.\n"
    "🚨 If you search and find tools but don't call them, you FAIL the task.\n"
    "🚨 After searching, your NEXT action MUST be calling one or more tools.\n\n"
    "═══ EXECUTION RULES ═══\n"
    "✓ Call ALL relevant tools found in your search (not just one)\n"
    "✓ Call each tool ONLY ONCE (no duplicates)\n"
    "✓ Use default parameters from tool descriptions when available\n"
    "✓ Extract parameters from query context when possible\n"
    "✓ If multiple tools are needed, call them all before responding\n\n"
    "═══ WHAT IS WRONG ═══\n"
    "❌ WRONG: Searching, then explaining which tools to use → FAIL\n"
    "❌ WRONG: Listing tools without calling them → FAIL\n"
    "❌ WRONG: Asking user to call tools → FAIL\n"
    "✅ CORRECT: Search → Call tools → Respond with results\n\n"
    "═══ PARAMETER HANDLING ═══\n"
    "When a tool needs parameters:\n"
    "1. Check tool description for defaults → USE THEM\n"
    "2. Extract from query context\n"
    "3. Use reasonable assumptions\n"
    "4. Only ask user if absolutely no other option\n\n"
    "Remember: Your job is to EXECUTE tools and provide results, not to explain what tools exist.\n"
)

TOOL_HUB_DESCRIPTION = (
    "Universal tool hub for dynamic tool discovery and execution.\n"
    "⚠️ THIS IS YOUR ONLY TOOL. You MUST search before calling any tools.\n\n"
    "🚨 CRITICAL: After searching, you MUST immediately call the relevant tools. Do NOT just list them.\n\n"
    "═══ TWO MODES ═══\n"
    "1️⃣ SEARCH MODE: Find tools\n"
    "   action='search', query='<specific task>', k=10\n"
    "   → Returns: List of available tools with descriptions\n\n"
    "2️⃣ CALL MODE: Execute tools (MANDATORY after searching)\n"
    "   action='call', tool_name='<exact_name>', tool_input='{\"param\": \"value\"}'\n"
    "   → Returns: Actual results from the tool\n\n"
    "═══ WORKFLOW ═══\n"
    "1. Search for tools → Get tool list\n"
    "2. Immediately call relevant tools → Get results\n"
    "3. Respond with results\n\n"
    "🚨 DO NOT: Search and then explain what you found without calling tools\n"
    "✅ DO: Search → Call tools → Respond with results\n\n"
    "═══ EXAMPLES ═══\n\n"
    "Example 1: Query needs multiple tools\n"
    "User: 'Get current status and full history for package ABC'\n"
    "Action 1: tool_hub(action='search', query='package tracking status history', k=10)\n"
    "Action 2: tool_hub(action='call', tool_name='get_latest_status', tool_input='{\"id\": \"ABC\"}')\n"
    "Action 3: tool_hub(action='call', tool_name='get_full_history', tool_input='{\"id\": \"ABC\"}')\n"
    "Response: [Provide results from both tools]\n\n"
    "Example 2: Using defaults from tool description\n"
    "User: 'Calculate the average temperature for this week'\n"
    "Action 1: tool_hub(action='search', query='weekly temperature average calculation', k=10)\n"
    "→ Tool description shows default: location='New York', days=7\n"
    "Action 2: tool_hub(action='call', tool_name='calculate_weekly_avg_temp', tool_input='{\"location\": \"New York\", \"days\": 7}')\n"
    "Response: [Provide calculation result]\n"
    "❌ WRONG: 'I need the location' → Should check defaults first!\n\n"
    "Example 3: Multiple different tools (NO duplicates!)\n"
    "User: 'Get weather forecast and traffic conditions for my commute'\n"
    "Action 1: tool_hub(action='search', query='weather forecast traffic conditions', k=10)\n"
    "Action 2: tool_hub(action='call', tool_name='get_weather_forecast', tool_input='{\"city\": \"Boston\"}')\n"
    "Action 3: tool_hub(action='call', tool_name='get_traffic_status', tool_input='{\"route\": \"I-95\"}')\n"
    "Response: [Provide combined results from both tools]\n"
    "❌ WRONG: Calling 'get_weather_forecast' multiple times instead of calling DIFFERENT tools!\n\n"
    "═══ KEY RULES ═══\n"
    "✅ After search → Immediately call relevant tools\n"
    "✅ Call multiple DIFFERENT tools when needed\n"
    "✅ Check tool descriptions for default parameters\n"
    "✅ Call each tool ONLY ONCE\n\n"
    "❌ DO NOT search and then just explain what you found\n"
    "❌ DO NOT call the same tool multiple times\n"
    "❌ DO NOT ask for parameters before checking defaults\n"
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
    - recursion_limit: Maximum iterations for LangGraph agent (default: 20)

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
                       retrieved_tools: List[str], search_state: Dict[str, int]) -> str:
        """Handle tool search action with search loop detection."""
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
        
        # Log search results with tool names
        retrieved_names = [t.name for t in matches]
        log_verbose(f"┌─ SEARCH: '{query}'")
        log_verbose(f"│  Retrieved {len(matches)} tools: {retrieved_names}")
        log_verbose(f"│  Newly added: {newly_added}")
        log_verbose(f"└─ Total unique retrieved: {len(retrieved_tools)}")

        # Track consecutive searches for loop detection
        search_state["count"] = search_state.get("count", 0) + 1
        consecutive = search_state["count"]
        
        # Detect search loop conditions
        no_new_tools = len(newly_added) == 0 and len(active_tools) > 0
        too_many_searches = consecutive >= MAX_CONSECUTIVE_SEARCHES
        
        # Build response payload
        payload = {
            "tools": tool_summaries,
            "next_action": "Use action='call' to execute the tools listed above. Do NOT search again.",
        }
        
        # Add warnings for search loop conditions
        if no_new_tools:
            payload["warning"] = (
                "⚠️ NO NEW TOOLS FOUND. You already have all relevant tools from previous searches. "
                "STOP SEARCHING IMMEDIATELY and CALL the tools you already found."
            )
            log_verbose(f"│  ⚠️ Search loop: no new tools found")
        
        if too_many_searches:
            payload["critical"] = (
                f"🚨 STOP: You have searched {consecutive} times without calling any tools. "
                "You MUST call tools NOW using action='call'. DO NOT search again or you will fail."
            )
            log_verbose(f"│  🚨 {consecutive} consecutive searches - forcing tool calls")
        
        # If search loop detected, change status to signal the model should stop
        status = "success"
        if no_new_tools or too_many_searches:
            status = "search_complete"
            payload["must_call_now"] = True

        return json.dumps({
            "mode": "search",
            "status": status,
            "payload": payload,
        })

    async def _handle_call(self, tool_name: str, tool_input: str, active_tools: List[BaseTool],
                           called_tools: List[str], lock: asyncio.Lock,
                           search_state: Dict[str, int]) -> str:
        """Handle tool invocation action with proper parameter validation and duplicate prevention."""
        
        # Reset consecutive search counter when a tool is called
        search_state["count"] = 0
        
        # Check if already successfully called (use lock for thread safety)
        async with lock:
            if tool_name in called_tools:
                return json.dumps({
                    "mode": "call",
                    "status": "already_called",
                    "payload": {
                        "error": f"Tool '{tool_name}' was already called successfully.",
                        "hint": "Do not call the same tool twice. Move on to the next step.",
                    }
                })
            # NOTE: We do NOT add to called_tools here anymore!
            # Only add after successful execution to allow retries on validation/execution failures
        
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

        # Note: [TOOL] logging is handled by log_tool decorator in mcp_proxy.py

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

        # Only mark as called AFTER successful execution (allows retries on failures)
        async with lock:
            if tool_name not in called_tools:
                called_tools.append(tool_name)

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

    def _make_tool_hub(self, active_tools: List[BaseTool], retrieved_tools: List[str],
                        called_tools: List[str], lock: asyncio.Lock) -> BaseTool:
        """
        Create the single tool_hub tool for searching and calling other tools.

        The returned closure captures self for accessing instance state and the 
        query-scoped active_tools list. called_tools prevents duplicate calls.
        lock ensures atomic duplicate checking for parallel tool calls.
        """
        # Track consecutive searches to detect loops
        search_state: Dict[str, int] = {"count": 0}
        
        async def run(action: str = "", query: str = "", k: Optional[int] = None,
                      tool_name: str = "", tool_input: str = "") -> str:
            act = (action or "").strip().lower()

            if act in ("search", "find", "fetch") or (not act and query):
                return self._handle_search(query, k, active_tools, retrieved_tools, search_state)

            if act == "call" or tool_name:
                return await self._handle_call(tool_name, tool_input, active_tools, called_tools, lock, search_state)

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
        called_tools: List[str] = []  # Prevents duplicate tool calls
        lock = asyncio.Lock()  # Ensures atomic duplicate checking for parallel calls

        hub = self._make_tool_hub(active_tools, retrieved_tools, called_tools, lock)

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
        final_retrieved = self._dedup_keep_order(retrieved_tools)
        
        # Log final summary
        log_verbose(f"╔══ QUERY COMPLETE ══════════════════════════════════════")
        log_verbose(f"║  RETRIEVED TOOLS ({len(final_retrieved)}): {final_retrieved}")
        log_verbose(f"║  CALLED TOOLS ({len(called_tools)}): {called_tools}")
        log_verbose(f"╚═════════════════════════════════════════════════════════")
        
        return result, final_retrieved

    def tear_down(self) -> None:
        """Clean up resources."""
        self.tool_name_to_base_tool = None
        self.vector_store = None
        self.embeddings = None
        self.reranker = None