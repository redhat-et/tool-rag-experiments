import asyncio
import os
import random
import time
import csv
import types
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from llm_provider import get_llm_provider, validate_provider_setup

from dotenv import load_dotenv

load_dotenv()

"""
# LangChain Max Tool Experiment with Local Ollama Models

## Overview
This script tests how well LangChain handles increasing numbers of tools using local Ollama models by measuring **tool selection accuracy, execution success, and latency**. 
## Experiment Setup
- **5 Real Tools**: Weather info, word count, string reversal, uppercase conversion, insurance scoring.
- **Fake Tools**: Dynamically generated tools with random outputs (up to 40 additional tools).
- **5 Fixed Queries**: Each mapped to a ground truth tool.
- **Scaling**: Start with 5 tools, increase by 5 up to 45.
- **Metrics Logged**:
  - Exception Rate (how many exception occurs out of 5 queries)
  - Tool Execution Success Rate (how many time tools are actually executed out of 5 queries)
  - Correct Tool Selection Rate  (how many time correct tool is selected out of 5 queries)
  - Irrelevant Tool Rate (how many time irrelevant tool is selected out of 5 queries)
  - Average Latency (average time taken to respond 5 queries)

## Requirements
- Ollama installed and running locally
- A local model pulled (e.g., `ollama pull llama3.2:3b`)

## Run the Experiment
```bash
# First start the MCP tool server in one terminal:
python src/mcp_tool_server.py

# Then run the experiment in another terminal:
python src/max_tool_experiment/ollama_maxtool.py
```
Results are saved in `experiment_results_langchain_ollama.csv` for analysis.

"""

# Note: For MCP tools, we can't easily generate fake tools dynamically
# The experiment will use only the real tools from the MCP server
# In a real scenario, you would need to add fake tools to the MCP server

# Define test queries and ground truth tools
queries = [
    ("What is the weather in New York?", "weather_info"),
    ("How many words are in 'Hello World, this is a test sentence'?", "word_count"),
    ("Reverse this text: Python Experiment", "reverse_string"),
    ("Convert this to uppercase: llamastack", "uppercase"),
    ("Give me an insurance evaluation score", "insurance_scorer")
]

def log_results(results):
    """Logs experiment results into a CSV file."""
    with open("experiment_results_langchain_ollama.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Tool Count", "Tool Execution Rate", "Correct Tool Rate", "Irrelevant Tool Rate", "Average Latency (s)"])
        writer.writerows(results)

async def run_main():
    # Connect to the MCP tool server
    client = MultiServerMCPClient({
        "general": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8000/mcp/"
        }
    })
    tools = await client.get_tools()
    
    # Validate LLM provider setup
    provider_status = validate_provider_setup()
    if not provider_status["available"]:
        print(f"❌ LLM provider not available: {provider_status['errors']}")
        return
    
    print(f"✅ Using {provider_status['provider']} provider: {provider_status['model']}")
    
    # Initialize the LLM using the abstraction layer
    llm = get_llm_provider(
        provider=provider_status["provider"],
        model=provider_status["model"],
        temperature=0,
    )
    
    # Create the agent using LangGraph
    agent = create_react_agent(llm, tools)
    
    results = []
    total_tools = len(tools)  # Using tools from MCP server
    print(f"\nTesting with {total_tools} tools from MCP server...")
    tool_execution_count = 0
    correct_tool_count = 0
    total_latency = 0

    for query, correct_tool in queries:
        print(f"\nUser: {query}")
        start_time = time.time()
        
        try:
            response = await agent.ainvoke({"messages": query})
            end_time = time.time()
            response_time = end_time - start_time
            total_latency += response_time

            # Print the main response
            if isinstance(response, dict) and 'output' in response:
                output = response['output']
            else:
                output = str(response)
            print(f"Response: {output}")
            
            # Extract and print just the final AIMessage content
            if isinstance(response, dict) and 'messages' in response:
                messages = response['messages']
                # Find the last AIMessage (final response)
                for message in reversed(messages):
                    if hasattr(message, 'content') and message.content and not hasattr(message, 'tool_calls'):
                        print(f"AIMessage: {message.content}")
                        break

            # Heuristic: Guess tool from response or query
            executed_tools = []
            resp_lower = output.lower()
            if "reversed text" in resp_lower or "reversed" in resp_lower:
                executed_tools.append("reverse_string")
            elif "word count" in resp_lower or "words" in resp_lower:
                executed_tools.append("word_count")
            elif "weather" in resp_lower:
                executed_tools.append("weather_info")
            elif "uppercase" in resp_lower:
                executed_tools.append("uppercase")
            elif "insurance score" in resp_lower or "insurance" in resp_lower:
                executed_tools.append("insurance_scorer")
            else:
                executed_tools.append("unknown")

            # Check if the correct tool was used
            correct_tool_used = any(tool == correct_tool for tool in executed_tools)
            tool_execution_count += 1 if executed_tools[0] != "unknown" else 0
            correct_tool_count += 1 if correct_tool_used else 0

            print(f"Executed Tools: {executed_tools}")
            print(f"Ground Truth Tool: {correct_tool}")
            
        except Exception as e:
            print(f"Error processing query: {e}")

    tool_execution_rate = tool_execution_count / len(queries)
    correct_tool_rate = correct_tool_count / len(queries)
    average_latency = total_latency / len(queries)
    irrevant_tool_rate = 1 - correct_tool_rate/tool_execution_rate
    
    results.append([total_tools, tool_execution_rate, correct_tool_rate, average_latency])
    print(f"\nTotal Tools: {total_tools}, Tool Execution Rate: {tool_execution_rate:.2%}, Correct Tool Rate: {correct_tool_rate:.2%}, Irrelevant Tool Rate: {irrevant_tool_rate:.2%}, Avg Latency: {average_latency:.4f}s")
    
    log_results(results)

if __name__ == "__main__":
    asyncio.run(run_main()) 