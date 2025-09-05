#!/usr/bin/env python3
"""
FAC-Only Evaluation System
Simplified evaluation focusing only on Final Answer Correctness (FAC) metric.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import existing components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))
from llm_provider import get_llm

load_dotenv()

class FACOnlyEvaluation:
    """Evaluate LangGraph ReAct agent using FAC metric only."""
    
    def __init__(self):
        self.output_dir = Path("fac_evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_environment()
        
        # Load synthetic dataset
        self.synthetic_queries = self.load_synthetic_queries()
        
        # Initialize LLM first
        self.llm = self.initialize_llm()
        
        # Initialize tools and agent (will be set in setup_mcp_connection)
        self.tools = None
        self.agent = None
    
    async def setup_mcp_connection(self):
        """Connect to MCP server and setup agent."""
        # Connect to MCP server and get tools
        self.tools = await self.connect_to_mcp_tools()
        
        # Create agent with MCP tools using default LangGraph prompt
        self.agent = create_react_agent(self.llm, self.tools)
    
    def setup_environment(self):
        """Setup environment for FAC evaluation."""
        # Create output directories
        self.raw_dir = self.output_dir / "raw_answers" / "langgraph_react"
        self.converted_dir = self.output_dir / "converted_answers" / "langgraph_react"
        self.eval_dir = self.output_dir / "evaluation"
        
        for dir_path in [self.raw_dir, self.converted_dir, self.eval_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_synthetic_queries(self) -> List[Dict[str, Any]]:
        """Load synthetic queries from manual dataset."""
        dataset_file = Path("src/max_tool_experiment/synthetic_dataset/test_instruction/G1_instruction.json")
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, 'r') as f:
            queries = json.load(f)
        
        print(f"✅ Loaded {len(queries)} queries from manual dataset: {dataset_file}")
        return queries
    
    async def connect_to_mcp_tools(self):
        """Connect to MCP server and retrieve tools using MultiServerMCPClient."""
        try:
            print("🔌 Connecting to MCP server using MultiServerMCPClient...")
            
            # Use MultiServerMCPClient to connect to the MCP server
            client = MultiServerMCPClient({
                "general": {
                    "url": "http://127.0.0.1:8000/mcp/",
                    "transport": "streamable_http",
                }
            })
            
            # Get tools from the client
            tools = await client.get_tools()
            print(f"✅ Successfully retrieved {len(tools)} tools from MCP server")
            
            # Debug: Print tool details
            for i, tool in enumerate(tools):
                print(f"  Tool {i+1}: {tool.name} - {tool.description}")
                if hasattr(tool, 'args_schema'):
                    print(f"    Args: {tool.args_schema}")
            
            return tools
            
        except Exception as e:
            print(f"❌ Error connecting to MCP server: {e}")
            print("⚠️ Falling back to empty tools list")
            return []
    

    
    def initialize_llm(self):
        """Initialize LLM provider using existing component."""
        provider_id = os.getenv("LLM_PROVIDER", "openai")
        model_id = os.getenv("LLM_MODEL", "gpt-4.1-nano")
        base_url = os.getenv("LLM_BASE_URL", "")
        
        print(f"🤖 Using {provider_id} model: {model_id}")
        
        # Use existing get_llm function
        return get_llm(provider_id, model_id, base_url, temperature=0)
    
    async def run_complete_evaluation(self):
        """Run the complete FAC evaluation pipeline."""
        print("🚀 Starting FAC-Only Evaluation...")
        print("=" * 50)
        
        # Step 1: Setup MCP connection and agent
        await self.setup_mcp_connection()
        
        # Step 2: Run agent on all queries
        results = await self.run_agent_evaluation()
        
        # Step 3: Save results
        self.save_results(results)
        
        # Step 4: Run FAC evaluation
        fac_results = self.run_fac_evaluation()
        
        # Step 5: Print summary
        self.print_summary(results)
        
        # Step 6: Return metrics
        return {
            'total_queries': len(results),
            'solved_queries': len([r for r in results if r.get('success', False)]),
            'unsolved_queries': len([r for r in results if not r.get('success', False)]),
            'solve_rate': len([r for r in results if r.get('success', False)]) / len(results) * 100 if results else 0,
            'average_latency': sum(r.get('latency', 0) for r in results) / len(results) if results else 0,
            'total_latency': sum(r.get('latency', 0) for r in results),
            'raw_results': results,
            'converted_results_path': str(self.converted_dir / "G1_instruction.json"),
            'fac_evaluation_results': fac_results
        }
    
    async def run_agent_evaluation(self) -> List[Dict[str, Any]]:
        """Run the agent on all synthetic queries."""
        print("\n🤖 Running LangGraph ReAct agent on queries...")
        
        results = []
        total_latency = 0
        successful_queries = 0
        
        for i, query_data in enumerate(self.synthetic_queries, 1):
            query = query_data["query"]
            query_id = query_data.get("query_id", f"synthetic_{i:03d}")
            
            print(f"\n📝 Query {i}/{len(self.synthetic_queries)}: {query[:60]}...")
            
            start_time = time.time()
            
            try:
                # Run agent with detailed debugging
                print(f"🔧 Available tools: {[tool.name for tool in self.tools] if self.tools else 'No tools'}")
                print(f"🚀 Invoking agent with input: {query}")
                
                response = await self.agent.ainvoke({"messages": [{"role": "user", "content": query}]})
                
                print(f"📤 Raw agent response: {response}")
                print(f"📤 Response type: {type(response)}")
                if hasattr(response, 'keys'):
                    print(f"📤 Response keys: {list(response.keys())}")
                elif hasattr(response, '__dict__'):
                    print(f"📤 Response attributes: {list(response.__dict__.keys())}")
                
                latency = time.time() - start_time
                total_latency += latency
                
                result = {
                    "query_data": query_data,
                    "query_id": query_id,
                    "response": response,
                    "success": True,
                    "latency": latency,
                    "error": None
                }
                
                successful_queries += 1
                print(f"✅ Success (latency: {latency:.2f}s)")
                
            except Exception as e:
                latency = time.time() - start_time
                result = {
                    "query_data": query_data,
                    "query_id": query_id,
                    "response": None,
                    "success": False,
                    "latency": latency,
                    "error": str(e)
                }
                print(f"❌ Failed: {e}")
            
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results in FAC evaluation format."""
        print("\n💾 Saving results...")
        
        # Save raw format
        raw_results = []
        for result in results:
            if result["success"]:
                raw_format = self.convert_to_raw_format(
                    result["query_data"],
                    result["response"],
                    result["query_id"]
                )
                raw_results.append(raw_format)
        
        # Save individual raw files
        for raw_result in raw_results:
            query_id = raw_result["query_id"]
            filename = f"{query_id}_ReAct@1.json"
            filepath = self.raw_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(raw_result, f, indent=2)
        
        # Save converted format for FAC evaluation
        converted_results = {}
        for result in results:
            if result["success"]:
                converted_format = self.convert_to_fac_format(
                    result["query_data"],
                    result["response"],
                    result["query_id"]
                )
                converted_results[result["query_id"]] = converted_format
        
        # Save as G1_instruction.json
        converted_file = self.converted_dir / "G1_instruction.json"
        with open(converted_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"✅ Saved {len(raw_results)} raw results")
        print(f"✅ Saved {len(converted_results)} converted results")
    
    def convert_to_raw_format(self, query_data: Dict, response: Any, query_id: str) -> Dict:
        """Convert to raw format for storage."""
        # Convert LangGraph response to serializable format
        serializable_response = self.serialize_langgraph_response(response)
        
        return {
            "query_id": query_id,
            "query": query_data["query"],
            "response": serializable_response,
            "timestamp": time.time()
        }
    
    def serialize_langgraph_response(self, response: Any) -> Dict:
        """Convert LangGraph response to JSON-serializable format."""
        try:
            if hasattr(response, 'get'):
                # If response is a dict-like object
                return {
                    "output": str(response.get('output', '')),
                    "type": "dict_response"
                }
            elif hasattr(response, 'output'):
                # If response has output attribute
                return {
                    "output": str(response.output),
                    "type": "object_response"
                }
            else:
                # Fallback to string representation
                return {
                    "output": str(response),
                    "type": "string_response"
                }
        except Exception as e:
            return {
                "output": f"Error serializing response: {e}",
                "type": "error"
            }
    
    def convert_to_fac_format(self, query_data: Dict, response: Any, query_id: str) -> Dict:
        """Convert to FAC evaluation format."""
        # Extract final answer from response
        final_answer = self.extract_final_answer_from_response(response)
        
        return {
            "query": query_data["query"],
            "answer": {
                "final_answer": json.dumps({"final_answer": final_answer})
            }
        }
    
    def extract_final_answer_from_response(self, response: Any) -> str:
        """Extract final answer from LangGraph response."""
        try:
            print(f"🔍 Extracting final answer from response type: {type(response)}")
            
            # LangGraph returns {'messages': [...]} - we need to find the last AIMessage
            if isinstance(response, dict) and 'messages' in response:
                messages = response['messages']
                print(f"📨 Found {len(messages)} messages")
                
                # Look for the last AIMessage with the final answer (skip HumanMessage and ToolMessage)
                for i, msg in enumerate(reversed(messages)):
                    if hasattr(msg, 'content') and msg.content and hasattr(msg, '__class__'):
                        msg_class = msg.__class__.__name__
                        print(f"🔍 Message {len(messages)-i} ({msg_class}): {msg.content[:100]}...")
                        
                        # Look for AIMessage with actual content (not tool calls)
                        if msg_class == 'AIMessage' and msg.content and not msg.content.startswith('**'):
                            final_answer = msg.content.strip()
                            print(f"✅ Extracted final answer from AIMessage: {final_answer[:100]}...")
                            return final_answer
                        
                        # Also check for the old Action Input format as fallback
                        if '**Action Input**:' in msg.content:
                            # Extract the JSON part after Action Input
                            action_input_start = msg.content.find('**Action Input**:') + len('**Action Input**:')
                            action_input_text = msg.content[action_input_start:].strip()
                            
                            print(f"🎯 Found Action Input: {action_input_text[:100]}...")
                            
                            # Try to parse the JSON
                            try:
                                import json
                                action_data = json.loads(action_input_text)
                                if 'final_answer' in action_data:
                                    final_answer = action_data['final_answer']
                                    print(f"✅ Extracted final answer from Action Input: {final_answer[:100]}...")
                                    return final_answer
                            except json.JSONDecodeError as e:
                                print(f"⚠️ JSON decode error: {e}")
                                # Fallback: extract text between quotes after "final_answer"
                                if '"final_answer":' in action_input_text:
                                    start = action_input_text.find('"final_answer":') + len('"final_answer":')
                                    start = action_input_text.find('"', start) + 1
                                    end = action_input_text.find('"', start)
                                    if start > 0 and end > start:
                                        final_answer = action_input_text[start:end]
                                        print(f"✅ Extracted final answer (fallback): {final_answer[:100]}...")
                                        return final_answer
            
            # Fallback: try the old method
            if hasattr(response, 'get'):
                final_answer = response.get('output', '')
            elif hasattr(response, 'output'):
                final_answer = response.output
            else:
                final_answer = str(response)
            
            print(f"⚠️ Using fallback extraction: {final_answer[:100]}...")
            return str(final_answer)
            
        except Exception as e:
            print(f"❌ Error extracting final answer: {e}")
            import traceback
            traceback.print_exc()
            return str(response)
    
    def run_fac_evaluation(self):
        """Run FAC evaluation using StableToolBench."""
        print("\n🔍 Running StableToolBench FAC evaluation...")
        
        try:
            # Import and run StableToolBench FAC evaluation
            from metric_collectors.stabletoolbench_fac_eval import run_stabletoolbench_fac_eval
            
            success = run_stabletoolbench_fac_eval()
            
            if success:
                print("✅ StableToolBench FAC evaluation completed successfully!")
                return {"success": True, "results_path": "fac_evaluation_results/evaluation/fac_results.csv"}
            else:
                print("❌ StableToolBench FAC evaluation failed!")
                return {"success": False, "error": "FAC evaluation failed"}
                
        except Exception as e:
            print(f"❌ Error running StableToolBench FAC evaluation: {e}")
            return {"success": False, "error": str(e)}
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 FAC-ONLY EVALUATION SUMMARY")
        print("=" * 60)
        
        total_queries = len(results)
        successful_queries = sum(1 for r in results if r["success"])
        success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        
        print(f"📝 Total Queries: {total_queries}")
        print(f"✅ Successful: {successful_queries}")
        print(f"❌ Failed: {total_queries - successful_queries}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   - Raw results: {self.raw_dir}")
        print(f"   - Converted results: {self.converted_dir}")
        print(f"   - FAC evaluation: {self.eval_dir}")
        
        print("\n🎯 FAC-Only Evaluation Complete!")

async def main():
    """Main function to run the FAC-only evaluation."""
    evaluation = FACOnlyEvaluation()
    await evaluation.run_complete_evaluation()

if __name__ == "__main__":
    asyncio.run(main())
