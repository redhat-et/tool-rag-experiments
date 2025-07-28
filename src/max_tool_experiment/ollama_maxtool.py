import asyncio
import os
import random
import time
import csv
import types
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
  - Average Latency (average time taken to respond 5 queries)

## Requirements
- Ollama installed and running locally
```
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```

## Run the Experiment
```bash
python ollama_maxtool.py
```
Results are saved in `experiment_results_langchain_ollama.csv` for analysis.

"""

@tool
def weather_info(loc: str) -> str:
    """Fetches the current weather for a given location.
    
    Args:
        loc: The location for which weather information is requested.
        
    Returns:
        A string containing the weather information.
    """
    return f"Weather in {loc} is sunny."

@tool
def word_count(text: str) -> str:
    """Counts the number of words in the given text.
    
    Args:
        text: The input text to analyze.
        
    Returns:
        A string containing the word count.
    """
    return f"Word count: {len(text.split())}"

@tool
def reverse_string(text: str) -> str:
    """Reverses the given string.
    
    Args:
        text: The input text to reverse.
        
    Returns:
        A string containing the reversed text.
    """
    return f"Reversed text: {text[::-1]}"

@tool
def uppercase(text: str) -> str:
    """Converts the given string to uppercase.
    
    Args:
        text: The input text to convert.
        
    Returns:
        A string containing the uppercase text.
    """
    return f"Uppercase text: {text.upper()}"

@tool
def insurance_scorer() -> str:
    """Generates a random insurance score between 1 and 100.
    
    Returns:
        A string containing the generated random number.
    """
    return f"Insurance score: {random.randint(1, 100)}"

def generate_random_text(length=10):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu"]
    return "_".join(random.choices(words, k=length))

# Generate fake tools using `types.FunctionType`
def generate_fake_tools(n):
    tools = []
    
    for i in range(n):
        tool_name = f"tool_{i}_{generate_random_text(2)}"
        tool_doc = f"""Tool {i} performs a unique operation on the input data. {generate_random_text(5)}
        
        Args:
            input_data: The input data for the tool.
            
        Returns:
            A string with an irrelevant response.
        """
        
        def fake_tool(input_data: str, tool_id=i):
            return f"Fake Tool {tool_id} received input: {input_data}"
        
        fake_tool_fn = types.FunctionType(fake_tool.__code__, globals(), tool_name)
        fake_tool_fn.__doc__ = tool_doc
        fake_tool_fn = tool(fake_tool_fn)
        
        tools.append(fake_tool_fn)
    
    return tools

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
        writer.writerow(["Tool Count", "Exception Rate", "Tool Execution Rate", "Correct Tool Rate", "Average Latency (s)"])
        writer.writerows(results)

async def run_main():
    # Initialize the local Ollama model
    # Available models: llama3.2:3b-instruct-fp16 (6.4GB), llama3.2:1b-instruct-fp16 (2.5GB)
    llm = ChatOllama(
        model="llama3.2:3b-instruct-fp16",  # Using available model
        base_url="http://localhost:11434",
        temperature=0,
    )
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant with access to many internal tools. Must use one correct internal tool for each query.
        When using the tools:
        1. Extract the relevant number or values from the user's request.
        2. Must use one correct tool to perform the operation.
        3. Present the tool call results concisely."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    real_tools = [weather_info, word_count, reverse_string, uppercase, insurance_scorer]
    results = []

    for total_tools in range(5, 50, 5):  # Increase by 5 up to 45 tools, can switch to step size 1.
        tools = real_tools + generate_fake_tools(total_tools - len(real_tools))
        
        # Create the agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=3
        )
        
        print(f"\nTesting with {total_tools} tools...")
        exception_count = 0
        tool_execution_count = 0
        correct_tool_count = 0
        total_latency = 0

        for query, correct_tool in queries:
            print(f"\nUser: {query}")
            start_time = time.time()
            
            try:
                response = await agent_executor.ainvoke({"input": query})
                end_time = time.time()
                response_time = end_time - start_time
                total_latency += response_time

                print(f"Response: {response['output']}")

                # Check if any tools were executed
                intermediate_steps = response.get('intermediate_steps', [])
                if intermediate_steps:
                    tool_executed = True
                    tool_execution_count += 1
                    
                    # Check if the correct tool was used
                    executed_tools = []
                    for step in intermediate_steps:
                        if hasattr(step[0], 'tool'):
                            executed_tools.append(step[0].tool)
                        elif hasattr(step[0], 'tool_name'):
                            executed_tools.append(step[0].tool_name)
                    
                    correct_tool_used = any(tool == correct_tool for tool in executed_tools)
                    correct_tool_count += correct_tool_used
                    
                    print(f"Executed Tools: {executed_tools}")
                    print(f"Ground Truth Tool: {correct_tool}")
                else:
                    print("No tools were executed")
                    print(f"Response keys: {response.keys()}")
                    print(f"Response type: {type(response)}")
                
            except Exception as e:
                print(f"Error processing query: {e}")
                exception_count += 1

        exception_rate = exception_count / len(queries)
        tool_execution_rate = tool_execution_count / len(queries)
        correct_tool_rate = correct_tool_count / len(queries)
        average_latency = total_latency / len(queries)
        
        results.append([total_tools, exception_rate, tool_execution_rate, correct_tool_rate, average_latency])
        print(f"\nTotal Tools: {total_tools}, Exception Rate: {exception_rate:.2%}, Tool Execution Rate: {tool_execution_rate:.2%}, Correct Tool Rate: {correct_tool_rate:.2%}, Avg Latency: {average_latency:.4f}s")
        if correct_tool_rate <0.5:
            break
    log_results(results)

if __name__ == "__main__":
    asyncio.run(run_main()) 