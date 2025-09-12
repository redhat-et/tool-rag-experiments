import asyncio
import sys
import os
sys.path.append('/Users/eoconnor/small-model-experiments')
from mcp_proxy import run_mcp_proxy
from mcp.server.fastmcp import FastMCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from evaluator.components.llm_provider import get_llm
from dotenv import load_dotenv

load_dotenv()

# Set your LLM provider details here
PROVIDER_ID = "ollama"  # or "vllm" or "openai"
MODEL_ID = "llama3.2:3b-instruct-fp16"  # or your model name
BASE_URL = "http://localhost:11434"  # or your LLM base URL


async def main():
    # Get MirrorAPI URL from environment variable or use default
    mirror_api_url = os.getenv("MIRROR_API_BASE_URL", "http://localhost:8000")

    # Load example_tools from JSON file
    import json
    with open("/Users/eoconnor/correct/small-model-experiments/evaluator/components/mcp_proxy/example_tools.json", "r") as f:
        example_tools = json.load(f)

    # Start MCP server and register proxy tools from the example list
    import threading
    import asyncio
    mcp = await run_mcp_proxy(example_tools, mcp_port=9000, server_name="General", mirror_api_base_url=mirror_api_url, run_detached=False)
    threading.Thread(target=lambda: mcp.run(transport="streamable-http"), daemon=True).start()
    await asyncio.sleep(2)  # Give server time to start

    # Connect to the MCP server
    client = MultiServerMCPClient({
        "general": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:9000/mcp/"
        }
    })
    tools = await client.get_tools()
    print("Available tools on MCP server (port 9000):")
    for tool in tools:
        print(f"- {tool.name}")

    # Initialize the LLM using the new get_llm signature
    llm = get_llm(PROVIDER_ID, MODEL_ID, BASE_URL, temperature=0)

    # Create the agent using LangGraph
    agent = create_react_agent(llm, tools)

    # Make a list of simple queries for each tool
    queries = [
        f"What does the tool '{tool.name}' do?" for tool in tools
    ]

    # Send each query to the agent and print the response
    for query in queries:
        print(f"\nQuery: {query}")
        response = await agent.ainvoke({"messages": query})
        print("Response:")
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
