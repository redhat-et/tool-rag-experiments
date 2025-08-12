"""
Enhanced Max Tool Experiment with ToolBench-style evaluation metrics.
This script integrates ToolBench's evaluation methodology into the existing max tool experiment.
All tools (including any StableToolBench tools) come from the MCP server.
We simply add ToolBench evaluation on top of the existing tool execution.
"""

import asyncio
import time
# import csv
import json
from typing import List, Dict, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from llm_provider import _get_provider, validate_provider_setup
from simplified_toolbench_evaluator import SimplifiedToolBenchEvaluator, PassRateResult
from dotenv import load_dotenv

load_dotenv()

class EnhancedToolLogger:
    def __init__(self, log_file="tool_log.txt"):
        self.log_file = log_file
        self.execution_details = []
    
    def get_executed_tools(self):
        """Read tool names from the log file."""
        executed_tools = []
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    if line.strip().startswith("[TOOL]"):
                        tool_name = line.strip().split("[TOOL] ")[1]
                        executed_tools.append(tool_name)
        except FileNotFoundError:
            pass
        self.clear_log()
        return executed_tools
    
    def clear_log(self):
        """Clear the log file."""
        try:
            with open(self.log_file, "w") as f:
                f.write("")
            print("✅ tool_log.txt cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing tool_log.txt: {e}")
    
    def add_execution_detail(self, detail: Dict[str, Any]):
        """Add execution detail for ToolBench-style evaluation."""
        self.execution_details.append(detail)

# --- Helper: Extract final answer text from agent response ---
def extract_final_answer_from_response(response: Any) -> str:
    """Best-effort extraction of the assistant's final text from a LangGraph agent response."""
    try:
        # If response is already a string
        if isinstance(response, str):
            return response

        # If dict-like
        if isinstance(response, dict):
            # Common key used by some agents
            out = response.get("output")
            if isinstance(out, str) and out.strip():
                return out

            # LangGraph prebuilt often returns messages
            messages = response.get("messages")
            if isinstance(messages, list) and messages:
                # Find the last message with textual content
                for msg in reversed(messages):
                    # Message as dict with 'content' or 'text'
                    if isinstance(msg, dict):
                        content = msg.get("content") or msg.get("text")
                        if isinstance(content, str) and content.strip():
                            return content
                        # Some message contents are lists of parts
                        if isinstance(content, list) and content:
                            # Join string parts if present
                            parts = [p for p in content if isinstance(p, str)]
                            if parts:
                                return "\n".join(parts)
                    else:
                        # Object-style message with attribute 'content'
                        content = getattr(msg, "content", None)
                        if isinstance(content, str) and content.strip():
                            return content

            # Some variants nest a return value
            rv = response.get("return_values")
            if isinstance(rv, dict):
                text = rv.get("output") or rv.get("final_answer")
                if isinstance(text, str) and text.strip():
                    return text

        # Object with attribute 'output' or 'content'
        content = getattr(response, "output", None) or getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    except Exception:
        pass

    return ""

# Enhanced queries with expected tools and parameters for ToolBench-style evaluation
# These work with any tools available in the MCP server (including StableToolBench tools)
# To add StableToolBench queries, simply add them to this list with their expected tools
ENHANCED_QUERIES = [
    {
        "query": "What is the weather in New York?",
        "expected_tool": "weather_info",
        "expected_params": {"loc": "New York"},
        "category": "weather"
    },
    {
        "query": "Count the words in 'Hello world'",
        "expected_tool": "word_count",
        "expected_params": {"text": "Hello World"},
        "category": "text_analysis"
    },
    {
        "query": "Reverse this text: Python Experiment",
        "expected_tool": "reverse_string",
        "expected_params": {"text": "Python Experiment"},
        "category": "text_manipulation"
    },
    {
        "query": "Convert this to uppercase: llamastack",
        "expected_tool": "uppercase",
        "expected_params": {"text": "llamastack"},
        "category": "text_manipulation"
    },
    {
        "query": "Give me an insurance evaluation score for a 25 year old male with a history of smoking",
        "expected_tool": "insurance_scorer",
        "expected_params": {"text": "25 year old male with a history of smoking"},
        "category": "scoring"
    }
]

def extract_params_from_query(query: str, expected_tool: str) -> Dict[str, Any]:
    """Extract parameters from query based on expected tool."""
    params = {}
    
    if expected_tool == "weather_info":
        # Extract location from weather queries
        if "weather" in query.lower():
            # Simple extraction - in real implementation, use more sophisticated parsing
            if "in" in query:
                location = query.split("in")[-1].strip().rstrip("?")
                params["loc"] = location
    
    elif expected_tool == "word_count":
        # Extract text from word count queries
        if "words are in" in query:
            text_start = query.find("'") + 1
            text_end = query.rfind("'")
            if text_start > 0 and text_end > text_start:
                params["text"] = query[text_start:text_end]
    
    elif expected_tool == "reverse_string":
        # Extract text from reverse queries
        if "reverse this text:" in query.lower():
            text_part = query.split("reverse this text:")[-1].strip()
            params["text"] = text_part
    
    elif expected_tool == "uppercase":
        # Extract text from uppercase queries
        if "convert this to uppercase:" in query.lower():
            text_part = query.split("convert this to uppercase:")[-1].strip()
            params["text"] = text_part
    
    return params

def log_enhanced_results(result: Dict[str, Any], filename: str = "enhanced_experiment_results.json"):
    """Save a single JSON dict with pass-rate and agent metrics."""
    with open(filename, "w") as f:
        json.dump(result, f, indent=2, default=str)

async def run_enhanced_experiment():
    """Run the enhanced experiment with ToolBench-style evaluation."""
    
    # Initialize Simplified ToolBench pass-rate evaluator (no Finish step validation)
    pass_rate_evaluator = SimplifiedToolBenchEvaluator(evaluate_times=4, max_eval_threads=4)
    tool_logger = EnhancedToolLogger()

    # Counters for agent metrics
    tool_execution_count = 0  # queries where at least one tool executed
    correct_tool_count = 0
    irrelevant_tool_count = 0
    total_latency = 0.0
    
    # Connect to the MCP tool server
    import os
    mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp/")
    client = MultiServerMCPClient({
        "general": {
            "transport": "streamable_http",
            "url": mcp_url
        }
    })
    tools = await client.get_tools()
    
    # Validate LLM provider setup
    provider_status = validate_provider_setup()
    if not provider_status["available"]:
        print(f"❌ LLM provider not available: {provider_status['errors']}")
        return
    
    print(f"✅ Using {provider_status['provider']} provider: {provider_status['model']}")
    
    # Get provider configuration
    provider_config = _get_provider()
    
    # Initialize the LLM based on provider
    if provider_config["provider_id"] == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=provider_config["model"] or "llama3.2:3b-instruct-fp16",
            base_url=provider_config["base_url"] or "http://localhost:11434",
            temperature=0
        )
    elif provider_config["provider_id"] == "vllm":
        from langchain_community.llms import VLLM
        llm = VLLM(
            model=provider_config["model"] or "meta-llama/Llama-2-7b-chat-hf",
            endpoint=provider_config["base_url"] or "http://localhost:8000/v1",
            trust_remote_code=True,
            max_new_tokens=512,
            top_p=0.95,
            temperature=0
        )
    elif provider_config["provider_id"] == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=provider_config["model"] or "gpt-3.5-turbo",
            base_url=provider_config["base_url"],
            temperature=0
        )
    else:
        raise Exception(f"Unsupported provider: {provider_config['provider_id']}")
    
    # Create the agent using LangGraph
    prompt = "You are a tool call assistant. Keep answer short and concise."
    agent = create_react_agent(llm, tools, prompt=prompt)
    
    total_tools = len(tools)
    print(f"\nTesting with {total_tools} tools from MCP server...")
    
    for i, query_data in enumerate(ENHANCED_QUERIES):
        query = query_data["query"]
        expected_tool = query_data["expected_tool"]
        expected_params = query_data["expected_params"]
        
        print(f"\nUser: {query}")
        start_time = time.time()
        
        try:
            # Run agent
            response = await agent.ainvoke({"messages": query})
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Get executed tools from log file
            executed_tools = tool_logger.get_executed_tools()
            print(f"Executed Tools: {executed_tools}")
            print(f"Ground Truth Tool: {expected_tool}")
            
            actual_tool = executed_tools[0] if executed_tools else None
            actual_params = extract_params_from_query(query, actual_tool) if actual_tool else {}

            # --- Update counters for agent metrics ---
            if executed_tools:
                tool_execution_count += 1
                if expected_tool in executed_tools:
                    correct_tool_count += 1
                irrelevant_tool_count += len(executed_tools) - (1 if expected_tool in executed_tools else 0)

            total_latency += execution_time

            # Extract final answer text from response
            final_answer_text = extract_final_answer_from_response(response)

            # ToolBench Pass Rate Evaluation
            available_tool_names = [tool.name for tool in tools]
            pass_rate_result = pass_rate_evaluator.evaluate_query_pass_rate(
                query_id=f"query_{i}",
                query=query,
                expected_tool=expected_tool,
                available_tools=available_tool_names,
                agent_steps=executed_tools,
                final_answer=final_answer_text,
                execution_time=execution_time
            )
            print(f"ToolBench Pass Rate: {pass_rate_result.pass_rate:.1%}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
            
            # Create failed result
            result = {
                "query": query,
                "success": False
            }     
            
    # ---------- Compute & print agent metrics ----------
    num_queries = len(ENHANCED_QUERIES)
    tool_execution_rate = tool_execution_count / num_queries
    correct_tool_rate = correct_tool_count / num_queries
    irrelevant_tool_rate = irrelevant_tool_count / num_queries
    average_latency = total_latency / num_queries if num_queries else 0

    print(f"\n=== Agent Metrics (original experiment style) ===")
    print(f"Tool Execution Rate : {tool_execution_rate:.2%}")
    print(f"Correct Tool Rate   : {correct_tool_rate:.2%}")
    print(f"Irrelevant Tool Rate: {irrelevant_tool_rate:.2%}")
    print(f"Average Latency     : {average_latency:.3f}s")
    
    # Print ToolBench Pass Rate Results (Core Metric)
    pass_rate_evaluator.print_summary()
    
    # Generate and save comprehensive report
    pass_rate_report = pass_rate_evaluator.generate_toolbench_report()
    
    # Combine reports
    combined_report = {
        'toolbench_pass_rate_evaluation': pass_rate_report,
        'agent_metrics': {
            'tool_execution_rate': tool_execution_rate,
            'correct_tool_rate': correct_tool_rate,
            'irrelevant_tool_rate': irrelevant_tool_rate,
            'average_latency': average_latency
        },
        'experiment_config': {
            'total_tools': total_tools,
            'provider': provider_status['provider'],
            'model': provider_status['model'],
            'evaluation_methodology': 'ToolBench Pass Rate + Agent Performance'
        }
    }
    
    # Save results
    log_enhanced_results(combined_report, "enhanced_experiment_results.json")

if __name__ == "__main__":
    asyncio.run(run_enhanced_experiment()) 