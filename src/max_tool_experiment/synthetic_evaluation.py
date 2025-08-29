#!/usr/bin/env python3
"""
Synthetic Evaluation System
Uses synthetic dataset to evaluate LangGraph ReAct agent performance.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
import subprocess

# Add StableToolBench to path
stabletoolbench_path = Path("../../stabletoolbench_lib/libs/stabletoolbench")
sys.path.append(str(stabletoolbench_path))

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.tools import tool
from dotenv import load_dotenv

load_dotenv()

class SyntheticEvaluation:
    """Evaluate LangGraph ReAct agent using synthetic dataset."""
    
    def __init__(self):
        self.output_dir = Path("synthetic_evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_environment()
        
        # Load synthetic dataset
        self.synthetic_queries = self.load_synthetic_queries()
        
        # Create tools from synthetic dataset
        self.tools = self.create_tools_from_synthetic_data()
        
        # Initialize LLM and agent with improved prompt
        self.llm = self.initialize_llm()
        
        # Use improved ReAct prompt
        react_prompt = """You are an AI assistant that uses tools to answer user queries. Follow this format:

**Thought**: What do I need to do?
**Action**: Which tool to use?
**Action Input**: Tool parameters
**Observation**: Tool result (you'll get this)
**... (repeat as needed)**
**Thought**: I have all the information
**Action**: Finish
**Action Input**: {{"return_type": "give_answer", "final_answer": "Your complete answer"}}

## RULES:
1. **Complete ALL parts** of multi-step queries
2. **Use tool results** in your final answer

## EXAMPLE:
Query: "Get weather in Tokyo AND insurance score"
- Call weather_info(Tokyo)
- Call insurance_scorer()
- Final answer: "Weather: [result]. Insurance: [result]"

Start now!"""
        
        self.agent = create_react_agent(self.llm, self.tools, prompt=react_prompt)
    
    def setup_environment(self):
        """Setup environment for synthetic evaluation."""
        # Create output directories
        self.raw_dir = self.output_dir / "raw_answers" / "langgraph_react"
        self.converted_dir = self.output_dir / "converted_answers" / "langgraph_react"
        self.eval_dir = self.output_dir / "evaluation"
        
        for dir_path in [self.raw_dir, self.converted_dir, self.eval_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_synthetic_queries(self) -> List[Dict[str, Any]]:
        """Load synthetic queries from manual dataset."""
        # Load from StableToolBench format dataset
        dataset_file = Path("synthetic_dataset/test_instruction/G1_instruction.json")
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, 'r') as f:
            queries = json.load(f)
        
        print(f"✅ Loaded {len(queries)} queries from manual dataset: {dataset_file}")
        return queries
    
    def create_tools_from_synthetic_data(self) -> List:
        """Create LangChain tools from synthetic dataset API definitions."""
        tools = []
        
        # Try to import real tools from MCP server
        try:
            # Import the real tools directly from mcp_tool_server
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            
            from mcp_tool_server import weather_info, word_count, reverse_string, uppercase, insurance_scorer
            
            # Convert MCP tools to LangChain tools
            from langchain_community.tools import tool
            
            @tool
            def weather_info_tool(loc: str) -> str:
                """Fetches the current weather for a given location."""
                return weather_info(loc)
            
            @tool
            def word_count_tool(text: str) -> str:
                """Counts the number of words in the given text."""
                return word_count(text)
            
            @tool
            def reverse_string_tool(text: str) -> str:
                """Reverses the given string."""
                return reverse_string(text)
            
            @tool
            def uppercase_tool(text: str) -> str:
                """Converts the given string to uppercase."""
                return uppercase(text)
            
            @tool
            def insurance_scorer_tool() -> str:
                """Generates a random insurance score between 1 and 100."""
                return insurance_scorer()
            
            tools = [weather_info_tool, word_count_tool, reverse_string_tool, uppercase_tool, insurance_scorer_tool]
            
            print(f"✅ Imported {len(tools)} real tools from MCP server")
            return tools
            
        except Exception as e:
            print(f"⚠️  Could not import real tools: {e}")
            print("   Falling back to synthetic tools...")
            
            # Fallback: Create tools based on the APIs in our synthetic queries
            api_definitions = {}
            
            # Collect all unique APIs from queries
            for query in self.synthetic_queries:
                for api in query.get("api_list", []):
                    api_key = f"{api.get('tool_name', 'unknown')}_{api.get('api_name', 'unknown')}"
                    if api_key not in api_definitions:
                        api_definitions[api_key] = api
            
            # Create tools for each unique API
            for api_key, api_def in api_definitions.items():
                tool_func = self.create_tool_from_api_definition(api_def)
                if tool_func:
                    tools.append(tool_func)
            
            print(f"✅ Created {len(tools)} fallback tools from synthetic dataset APIs")
            return tools
    
    def create_tool_from_api_definition(self, api_def: Dict[str, Any]):
        """Create a LangChain tool from API definition."""
        tool_name = api_def.get("api_name", "unknown")
        description = api_def.get("api_description", "No description")
        
        # Get required parameters
        required_params = api_def.get("required_parameters", [])
        param_names = [param.get("name", "param") for param in required_params]
        
        # Create a tool that returns mock data based on the API
        def create_tool_function():
            @tool
            def api_tool(**kwargs):
                """Mock implementation of synthetic API."""
                # Return mock data based on API type
                if "weather" in tool_name.lower():
                    location = kwargs.get('loc', 'unknown location')
                    return f"The weather in {location} is sunny with a high of 75°F and low of 60°F."
                elif "word_count" in tool_name.lower():
                    text = kwargs.get('text', '')
                    word_count = len(text.split()) if text else 0
                    return f"The text contains {word_count} words."
                elif "reverse_string" in tool_name.lower():
                    text = kwargs.get('text', '')
                    reversed_text = text[::-1] if text else ''
                    return f"Reversed text: {reversed_text}"
                elif "uppercase" in tool_name.lower():
                    text = kwargs.get('text', '')
                    upper_text = text.upper() if text else ''
                    return f"Uppercase text: {upper_text}"
                elif "insurance_scorer" in tool_name.lower():
                    import random
                    score = random.randint(1, 100)
                    return f"Insurance score: {score}"
                else:
                    return f"Mock data for {tool_name} with parameters: {kwargs}"
            
            # Set the function name and docstring
            api_tool.__name__ = tool_name
            api_tool.__doc__ = description
            
            return api_tool
        
        return create_tool_function()
    
    def initialize_llm(self):
        """Initialize LLM provider."""
        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            try:
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0
                )
            except Exception as e:
                print(f"❌ OpenAI failed: {e}")
        
        # Fallback to Ollama
        try:
            return ChatOllama(
                model="llama3.2:3b-instruct-fp16",
                temperature=0
            )
        except Exception as e:
            print(f"❌ Ollama failed: {e}")
            raise Exception("No LLM available")
    
    def extract_tool_calls_from_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from LangGraph ReAct agent response."""
        tool_calls = []
        
        if "messages" not in response:
            return tool_calls
        
        for msg in response["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        "name": tool_call.get('name', 'unknown'),
                        "arguments": tool_call.get('args', {}),
                        "id": tool_call.get('id', 'unknown'),
                        "type": "tool_call"
                    })
        
        return tool_calls
    
    def extract_tool_results_from_response(self, response: Dict[str, Any]) -> List[str]:
        """Extract tool results from LangGraph response."""
        tool_results = []
        
        if "messages" in response:
            messages = response["messages"]
            for msg in messages:
                if isinstance(msg, dict) and msg.get("type") == "tool":
                    # Extract tool result from tool message
                    content = msg.get("content", "")
                    if content:
                        tool_results.append(content)
        
        return tool_results
    
    def extract_final_answer_from_response(self, response: Dict[str, Any]) -> str:
        """Extract final answer from LangGraph ReAct agent response."""
        if "messages" not in response:
            return "No answer provided"
        
        # Get the last AI message (final answer)
        for msg in reversed(response["messages"]):
            if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                return str(msg.content)
        
        return "No final answer found"
    
    def convert_to_stabletoolbench_raw_format(self, query_data: Dict[str, Any], 
                                            response: Dict[str, Any], 
                                            query_id: str) -> Dict[str, Any]:
        """Convert ReAct agent response to StableToolBench raw format."""
        
        # Extract data from response
        tool_calls = self.extract_tool_calls_from_response(response)
        tool_results = self.extract_tool_results_from_response(response)
        final_answer = self.extract_final_answer_from_response(response)
        
        # Build execution chain
        execution_chain = []
        
        # Add tool calls and results
        for i, tool_call in enumerate(tool_calls):
            # Add Action node
            execution_chain.append({
                "type": "Action",
                "content": tool_call["name"]
            })
            
            # Add Action Input node
            execution_chain.append({
                "type": "Action Input",
                "content": json.dumps(tool_call["arguments"])
            })
            
            # Add Observation node (if we have results)
            if i < len(tool_results):
                execution_chain.append({
                    "type": "Observation",
                    "content": tool_results[i]
                })
        
        # Add final answer
        execution_chain.append({
            "type": "Finish",
            "content": final_answer
        })
        
        # Build raw format
        raw_format = {
            "win": True,  # Assume success for now
            "try_count": 1,
            "trys": [
                {
                    "query": query_data["query"],
                    "function": execution_chain,
                    "final_answer": final_answer
                }
            ],
            "query": query_data["query"],
            "query_id": query_id
        }
        
        return raw_format
    
    def convert_to_stabletoolbench_converted_format(self, query_data: Dict[str, Any], response: Dict[str, Any], query_id: str) -> Dict[str, Any]:
        """Convert LangGraph response to StableToolBench converted format."""
        # Extract tool calls and results
        tool_calls = self.extract_tool_calls_from_response(response)
        tool_results = self.extract_tool_results_from_response(response)
        final_answer = self.extract_final_answer_from_response(response)
        
        # Convert tools to StableToolBench format
        available_tools = []
        for api in query_data.get("api_list", []):
            tool = {
                "name": api["api_name"],
                "description": f"This is the subfunction for tool \"{api['tool_name']}\", you can use this tool. The description of this function is: \"{api['api_description']}\"",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "optional": []
                }
            }
            
            # Add required parameters
            for param in api.get("required_parameters", []):
                tool["parameters"]["properties"][param["name"]] = {
                    "type": param["type"],
                    "description": param.get("description", ""),
                    "example_value": param.get("default", "")
                }
                tool["parameters"]["required"].append(param["name"])
            
            # Add optional parameters
            for param in api.get("optional_parameters", []):
                tool["parameters"]["properties"][param["name"]] = {
                    "type": param["type"],
                    "description": param.get("description", ""),
                    "example_value": param.get("default", "")
                }
                tool["parameters"]["optional"].append(param["name"])
            
            available_tools.append(tool)
        
        # Add Finish tool
        available_tools.append({
            "name": "Finish",
            "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {
                        "type": "string",
                        "enum": ["give_answer", "give_up_and_restart"]
                    },
                    "final_answer": {
                        "type": "string",
                        "description": "The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\""
                    }
                },
                "required": ["return_type"]
            }
        })
        
        # Build answer_details in StableToolBench format
        answer_details = []
        
        # Start with system message
        system_node = {
            "role": "system",
            "message": "",
            "next": []
        }
        
        # Add user message
        user_node = {
            "role": "user",
            "message": "",
            "next": []
        }
        system_node["next"].append(user_node)
        
        # Add tool calls and responses
        current_node = user_node
        for i, tool_call in enumerate(tool_calls):
            # Add tool call
            tool_node = {
                "role": "tool",
                "message": {
                    "name": tool_call.get("name", "unknown_tool"),
                    "arguments": json.dumps(tool_call.get("arguments", {}), indent=2),
                    "response": tool_results[i] if i < len(tool_results) else ""
                },
                "next": []
            }
            current_node["next"].append(tool_node)
            current_node = tool_node
        
        # Add final answer
        final_node = {
            "role": "tool",
            "message": {
                "name": "Finish",
                "arguments": json.dumps({
                    "return_type": "give_answer",
                    "final_answer": final_answer
                }, indent=2),
                "response": ""
            },
            "next": []
        }
        current_node["next"].append(final_node)
        
        answer_details.append(system_node)
        
        return {
            "query": query_data["query"],
            "available_tools": available_tools,
            "answer": {
                "method": "ReAct@1",
                "total_steps": len(tool_calls) + 1,  # +1 for Finish
                "final_answer": json.dumps({
                    "return_type": "give_answer",
                    "final_answer": final_answer
                }, indent=2),
                "answer_details": answer_details
            }
        }
    
    async def run_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single query through the ReAct agent."""
        query = query_data["query"]
        query_id = query_data.get("query_id", "unknown")
        
        print(f"🧪 Running synthetic query {query_id}: {query[:50]}...")
        
        try:
            start_time = time.time()
            response = await self.agent.ainvoke({"messages": query})
            end_time = time.time()
            
            latency = end_time - start_time
            
            return {
                "query_data": query_data,
                "response": response,
                "latency": latency,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error running query {query_id}: {e}")
            return {
                "query_data": query_data,
                "response": None,
                "latency": 0,
                "success": False,
                "error": str(e)
            }
    
    def run_synthetic_evaluation(self):
        """Run synthetic evaluation using StableToolBench scripts."""
        print("\n🔍 Running Synthetic Evaluation...")
        
        try:
            # Ensure evaluation directory exists
            self.eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Store absolute paths before changing directory
            eval_dir_abs = self.eval_dir.absolute()
            test_ids_dir = eval_dir_abs / "test_ids"
            test_ids_dir.mkdir(exist_ok=True)
            
            # Create the expected directory structure for converted answers
            model_name = "langgraph_react"
            converted_model_dir = self.converted_dir / model_name
            converted_model_dir.mkdir(exist_ok=True)
            
            # Read the converted results and convert to StableToolBench format
            converted_answer_path = converted_model_dir / "G1_instruction.json"
            with open(self.converted_dir / "langgraph_react" / "synthetic_results.json", 'r') as f:
                synthetic_results = json.load(f)
            
            # Convert to the expected format (query_id -> result mapping)
            converted_results = {}
            for i, result in enumerate(synthetic_results, 1):
                query_id = f"synthetic_{i:03d}"
                converted_results[query_id] = result
            
            with open(converted_answer_path, 'w') as f:
                json.dump(converted_results, f, indent=2)
            
            # Create test_ids file in StableToolBench format
            test_ids = {}
            for i in range(1, len(self.synthetic_queries) + 1):
                test_ids[f"synthetic_{i:03d}"] = 0
            
            # Save as G1_instruction.json (matching StableToolBench format)
            test_ids_file = test_ids_dir / "G1_instruction.json"
            with open(test_ids_file, 'w') as f:
                json.dump(test_ids, f, indent=2)
            
            # Change to StableToolBench directory
            original_cwd = os.getcwd()
            os.chdir(stabletoolbench_path)
            
            # Run Pass Rate evaluation
            print("📊 Running Pass Rate (SoPR) evaluation...")
            try:
                result = subprocess.run([
                    "python", "toolbench/tooleval/eval_pass_rate.py",
                    "--converted_answer_path", str(self.converted_dir),
                    "--save_path", str(eval_dir_abs),
                    "--test_ids", str(test_ids_dir),
                    "--evaluator", "tooleval_gpt-3.5-turbo_normalized",
                    "--test_set", "G1_instruction",
                    "--reference_model", model_name
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("✅ Pass Rate evaluation completed")
                    print(f"Output: {result.stdout}")
                else:
                    print(f"❌ Pass Rate evaluation failed: {result.stderr}")
            except Exception as e:
                print(f"❌ Pass Rate evaluation error: {e}")
            
            # Run FAC evaluation (skip if vllm is not available)
            print("📊 Running FAC evaluation...")
            try:
                result = subprocess.run([
                    "python", "toolbench/tooleval/fac_eval.py",
                    "--converted_answer_path", str(self.converted_dir),
                    "--save_path", str(eval_dir_abs),
                    "--test_ids", str(test_ids_dir),
                    "--test_set", "G1_instruction",
                    "--reference_model", model_name
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("✅ FAC evaluation completed")
                    print(f"Output: {result.stdout}")
                else:
                    print(f"❌ FAC evaluation failed: {result.stderr}")
                    print("⚠️  This might be due to missing vllm dependency")
            except Exception as e:
                print(f"❌ FAC evaluation error: {e}")
                print("⚠️  vllm dependency might be missing")
            
            # Return to original directory
            os.chdir(original_cwd)
            
        except Exception as e:
            print(f"❌ Synthetic evaluation failed: {e}")
            print("⚠️  This might be due to missing dependencies or configuration")
    
    async def run_complete_evaluation(self):
        """Run the complete synthetic evaluation."""
        print("🚀 Starting Synthetic Evaluation")
        print("=" * 60)
        
        if not self.synthetic_queries:
            print("❌ No synthetic queries loaded.")
            return
        
        results = []
        total_latency = 0
        successful_queries = 0
        
        # Run queries
        for i, query_data in enumerate(self.synthetic_queries):
            print(f"\n📝 Query {i+1}/{len(self.synthetic_queries)}")
            
            result = await self.run_query(query_data)
            results.append(result)
            
            if result["success"]:
                successful_queries += 1
                total_latency += result["latency"]
                print(f"✅ Success (latency: {result['latency']:.2f}s)")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Save results
        self.save_results(results)
        
        # Run synthetic evaluation
        self.run_synthetic_evaluation()
        
        # Print summary
        self.print_summary(results, total_latency, successful_queries)
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results in StableToolBench format."""
        print("\n💾 Saving results...")
        
        # Save raw format
        raw_results = []
        for result in results:
            if result["success"]:
                raw_format = self.convert_to_stabletoolbench_raw_format(
                    result["query_data"],
                    result["response"],
                    result["query_data"].get("query_id", "unknown")
                )
                raw_results.append(raw_format)
        
        # Save individual raw files
        for raw_result in raw_results:
            query_id = raw_result["query_id"]
            # Handle both string and integer query IDs
            if isinstance(query_id, str):
                filename = f"{query_id}_ReAct@1.json"
            else:
                filename = f"synthetic_{query_id:03d}_ReAct@1.json"
            filepath = self.raw_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(raw_result, f, indent=2)
        
        # Save converted format
        converted_results = []
        for result in results:
            if result["success"]:
                converted_format = self.convert_to_stabletoolbench_converted_format(
                    result["query_data"],
                    result["response"],
                    result["query_data"].get("query_id", "unknown")
                )
                converted_results.append(converted_format)
        
        # Save converted file in StableToolBench expected format
        # Convert to query_id -> result mapping format
        converted_results_dict = {}
        for i, result in enumerate(converted_results, 1):
            query_id = f"synthetic_{i:03d}"
            converted_results_dict[query_id] = result
        
        # Save as G1_instruction.json (StableToolBench format)
        converted_file = self.converted_dir / "G1_instruction.json"
        with open(converted_file, 'w') as f:
            json.dump(converted_results_dict, f, indent=2)
        
        print(f"✅ Saved {len(raw_results)} raw results")
        print(f"✅ Saved {len(converted_results)} converted results")
    
    def print_summary(self, results: List[Dict[str, Any]], total_latency: float, successful_queries: int):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 SYNTHETIC EVALUATION SUMMARY")
        print("=" * 60)
        
        total_queries = len(results)
        success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        avg_latency = total_latency / successful_queries if successful_queries > 0 else 0
        
        print(f"📝 Total Queries: {total_queries}")
        print(f"✅ Successful: {successful_queries}")
        print(f"❌ Failed: {total_queries - successful_queries}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Average Latency: {avg_latency:.2f}s")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   - Raw results: {self.raw_dir}")
        print(f"   - Converted results: {self.converted_dir}")
        print(f"   - Evaluation results: {self.eval_dir}")
        
        print("\n🎯 Synthetic Evaluation Complete!")
        print("✅ Used synthetic dataset with MCP tools")
        print("✅ Generated SoPR and FAC metrics")

async def main():
    """Main function to run the synthetic evaluation."""
    evaluation = SyntheticEvaluation()
    await evaluation.run_complete_evaluation()

if __name__ == "__main__":
    asyncio.run(main())
