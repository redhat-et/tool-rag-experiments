"""
ToolBench Pass Rate Evaluator for Max Tool Experiment
Core evaluation based on solvable pass rate from StableToolBench
"""

import json
import os
import statistics
from typing import Dict, List, Any, Tuple

import openai
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
class AnswerStatus(Enum):
    SOLVED = "AnswerStatus.Solved"
    UNSOLVED = "AnswerStatus.Unsolved" 
    UNSURE = "AnswerStatus.Unsure"

@dataclass
class PassRateResult:
    """Result of pass rate evaluation for a single query."""
    query_id: str
    query: str
    expected_tool: str
    available_tools: List[str]
    agent_steps: List[str]
    final_answer: str
    is_solved_evaluations: List[AnswerStatus]
    pass_rate: float
    execution_time: float

class ToolBenchPassRateEvaluator:
    """
    ToolBench-style pass rate evaluator for agent performance.
    Focuses on the core metric: solvable pass rate.
    """
    
    def __init__(self, evaluate_times: int = 4, judge_model: str = None):
        self.evaluate_times = evaluate_times
        self.results: List[PassRateResult] = []

        # OpenAI setup
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Please add it to your .env or environment.")

        self.judge_model = judge_model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-nano")

        

    def _judge_is_solved(self, query: str, answer: str) -> bool:
        """Call OpenAI using ToolBench's check_answer_status function to decide Solved vs Unsolved."""
        
        # ToolBench's original check_answer_status function template
        function_description = {
            "name": "check_answer_status",
            "description": """Giving the query and answer, you need give `answer_status` of the answer by following rules:
1. If the answer is a sorry message or not a positive/straight response for the given query, return "Unsolved".
2. If the answer is a positive/straight response for the given query, you have to further check.
2.1 If the answer is not sufficient to determine whether the solve the query or not, return "Unsure".
2.2 If you are confident that the answer is sufficient to determine whether the solve the query or not, return "Solved" or "Unsolved".

Query:
{query}
Answer:
{answer}

Now give your reason in "content" and `answer_status` of JSON to `check_answer_status`.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer_status": {
                        "type": "string",
                        "enum": ["Solved", "Unsolved", "Unsure"],
                        "description": "The status of whether the answer solves the query"
                    },
                    "content": {
                        "type": "string",
                        "description": "Explanation of the decision"
                    }
                },
                "required": ["answer_status", "content"]
            }
        }

        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Format the function description with the actual query and answer
            formatted_description = function_description["description"].format(
                query=query, 
                answer=answer
            )
            
            resp = client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "user", "content": formatted_description}
                ],
                tools=[{"type": "function", "function": function_description}],
                tool_choice={"type": "function", "function": {"name": "check_answer_status"}},
                temperature=0,
            )
            
            # Parse the function call response
            function_args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
            verdict = function_args["answer_status"]
            
        except Exception as e:
            # On failure, default to Unsolved
            print(f"⚠️ Judge error: {type(e).__name__}: {e}")
            verdict = "Unsolved"

        # Return True for "Solved", False for "Unsolved" or "Unsure"
        # Note: In ToolBench, "Unsure" counts as 0.5, but we're simplifying to binary for now
        return verdict == "Solved"
    
    def check_task_solvable(self, query: str, available_tools: List[str]) -> str:
        """Check if a task is solvable with the available tools using ToolBench's check_task_solvable function."""
        
        # ToolBench's original check_task_solvable function template
        function_description = {
            "name": "check_task_solvable",
            "description": """Please check whether the given task solvable with following rules:
1. If the `query` provide invalid information (e.g. invalid email address or phone number), return "Unsolvable"
2. If the `query` needs more information to solve (e.g. the target restaurant name in a navigation task), return "Unsolvable"
3. If you are unable to draw a conclusion, return "Unsure"
4. If the currently `available_tools` are enough to solve the query, return "Solvable"

Task:
{task}

Now give your reason in "content" and `task_status` of JSON to `check_task_solvable`.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_status": {
                        "type": "string",
                        "enum": ["Solvable", "Unsolvable", "Unsure"],
                        "description": "The status of whether the task is solvable"
                    },
                    "content": {
                        "type": "string",
                        "description": "Explanation of the decision"
                    }
                },
                "required": ["task_status", "content"]
            }
        }

        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Format the task description
            task_description = f"Query: {query}\nAvailable Tools: {', '.join(available_tools)}"
            formatted_description = function_description["description"].format(task=task_description)
            
            resp = client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "user", "content": formatted_description}
                ],
                tools=[{"type": "function", "function": function_description}],
                tool_choice={"type": "function", "function": {"name": "check_task_solvable"}},
                temperature=0,
            )
            
            # Parse the function call response
            function_args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
            status = function_args["task_status"]
            
        except Exception as e:
            # On failure, default to Unsure
            print(f"⚠️ Task solvability check error: {type(e).__name__}: {e}")
            status = "Unsure"

        return status
    
    def check_is_solved(self, query: str, expected_tool: str, available_tools: List[str], 
                       agent_steps: List[str], final_answer: str) -> Tuple[AnswerStatus, str]:
        """
        Core evaluation function that determines if a query is solved.
        Adapted from ToolBench's check_is_solved logic.
        """
        
        # Empty answer ⇒ Unsolved
        if not final_answer or final_answer.strip() == "":
            return AnswerStatus.UNSOLVED, "No final answer provided"

        # Expected tool must appear in the execution steps
        if expected_tool not in " ".join(agent_steps):
            return AnswerStatus.UNSOLVED, "Expected tool was not used"

        # Delegate judgement to OpenAI evaluator
        solved = self._judge_is_solved(query, final_answer)
        if solved:
            return AnswerStatus.SOLVED, "OpenAI judge marked as Solved"
        else:
            return AnswerStatus.UNSOLVED, "OpenAI judge marked as Unsolved"
    
    def evaluate_query_pass_rate(self, query_id: str, query: str, expected_tool: str,
                                available_tools: List[str], agent_steps: List[str], 
                                final_answer: str, execution_time: float) -> PassRateResult:
        """
        Evaluate a single query multiple times to calculate pass rate.
        Following ToolBench's evaluation methodology.
        """
        
        evaluations = []
        
        # Run evaluation multiple times for statistical significance
        for i in range(self.evaluate_times):
            is_solved, reason = self.check_is_solved(
                query, expected_tool, available_tools, agent_steps, final_answer
            )
            evaluations.append(is_solved)
        
        # Calculate pass rate
        solved_count = sum(1 for status in evaluations if status == AnswerStatus.SOLVED)
        unsure_count = sum(1 for status in evaluations if status == AnswerStatus.UNSURE)
        
        # ToolBench scoring: Solved = 1.0, Unsure = 0.5, Unsolved = 0.0
        pass_rate = (solved_count + (unsure_count * 0.5)) / len(evaluations)
        
        result = PassRateResult(
            query_id=query_id,
            query=query,
            expected_tool=expected_tool,
            available_tools=available_tools,
            agent_steps=agent_steps,
            final_answer=final_answer,
            is_solved_evaluations=evaluations,
            pass_rate=pass_rate,
            execution_time=execution_time
        )
        
        self.results.append(result)
        return result
    
    def calculate_overall_pass_rate(self) -> Dict[str, float]:
        """
        Calculate overall pass rate statistics following ToolBench methodology.
        """
        if not self.results:
            return {"pass_rate": 0.0, "std_dev": 0.0, "total_queries": 0}
        
        # Calculate pass rates for each evaluation run
        pass_rates_by_run = []
        for run_idx in range(self.evaluate_times):
            run_score = 0
            for result in self.results:
                status = result.is_solved_evaluations[run_idx]
                if status == AnswerStatus.SOLVED:
                    run_score += 1.0
                elif status == AnswerStatus.UNSURE:
                    run_score += 0.5
            
            pass_rates_by_run.append(run_score / len(self.results))
        
        # Calculate statistics
        overall_pass_rate = sum(pass_rates_by_run) / len(pass_rates_by_run)
        std_dev = statistics.stdev(pass_rates_by_run) if len(pass_rates_by_run) > 1 else 0.0
        
        return {
            "pass_rate": overall_pass_rate * 100,  # Convert to percentage
            "std_dev": std_dev * 100,  # Convert to percentage
            "total_queries": len(self.results),
            "pass_rates_by_run": [rate * 100 for rate in pass_rates_by_run]
        }
    
    def generate_toolbench_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report in ToolBench format."""
        stats = self.calculate_overall_pass_rate()
        
        # Detailed results per query
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                "query_id": result.query_id,
                "query": result.query,
                "expected_tool": result.expected_tool,
                "available_tools": result.available_tools,
                "agent_steps": result.agent_steps,
                "final_answer": result.final_answer,
                "is_solved_evaluations": [status.value for status in result.is_solved_evaluations],
                "pass_rate": result.pass_rate * 100,
                "execution_time": result.execution_time
            })
        
        return {
            "overall_statistics": stats,
            "detailed_results": detailed_results,
            "evaluation_config": {
                "evaluate_times": self.evaluate_times,
                "total_queries_evaluated": len(self.results)
            }
        }
    
    def save_results(self, filename: str):
        """Save results in ToolBench-compatible format."""
        report = self.generate_toolbench_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Print summary statistics following ToolBench format."""
        stats = self.calculate_overall_pass_rate()
        
        print(f"\n=== ToolBench Pass Rate Evaluation Results ===")
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Pass Rate: {stats['pass_rate']:.1f}%")
        print(f"Standard Deviation: {stats['std_dev']:.1f}%")
        
        # Print per-run results
        print(f"\nPass Rates by Evaluation Run:")
        for i, rate in enumerate(stats['pass_rates_by_run']):
            print(f"  Run {i+1}: {rate:.1f}%")


class MCPStableToolBenchEvaluator:
    """
    Evaluator specifically designed for MCP proxy with StableToolBench dataset.
    Prepares for integration with the upcoming MCP server implementation.
    """
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000/mcp/"):
        self.mcp_server_url = mcp_server_url
        self.pass_rate_evaluator = ToolBenchPassRateEvaluator()
    
    async def evaluate_mcp_toolbench_dataset(self, toolbench_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate agent performance on StableToolBench dataset via MCP proxy.
        
        Args:
            toolbench_queries: List of ToolBench queries in format:
                [{"query": str, "expected_tool": str, "query_id": str}, ...]
        """
        
        print(f"🔍 Evaluating {len(toolbench_queries)} ToolBench queries via MCP proxy...")
        
        results = []
        
        for i, query_data in enumerate(toolbench_queries):
            query_id = query_data.get("query_id", f"query_{i}")
            query = query_data["query"]
            expected_tool = query_data["expected_tool"]
            
            print(f"\n[{i+1}/{len(toolbench_queries)}] Evaluating: {query}")
            
            # This will be implemented when MCP proxy is ready
            # For now, we'll prepare the evaluation structure
            
            # Placeholder for MCP execution
            agent_steps = ["placeholder_step"]  # Will come from MCP agent execution
            final_answer = "placeholder_answer"  # Will come from MCP agent
            available_tools = ["placeholder_tools"]  # Will come from MCP server
            execution_time = 0.0  # Will be measured
            
            # Evaluate using pass rate evaluator
            result = self.pass_rate_evaluator.evaluate_query_pass_rate(
                query_id=query_id,
                query=query,
                expected_tool=expected_tool,
                available_tools=available_tools,
                agent_steps=agent_steps,
                final_answer=final_answer,
                execution_time=execution_time
            )
            
            results.append(result)
        
        # Generate final report
        report = self.pass_rate_evaluator.generate_toolbench_report()
        self.pass_rate_evaluator.print_summary()
        
        return report
    
    def prepare_toolbench_queries_from_mcp(self, mcp_dataset_path: str) -> List[Dict[str, Any]]:
        """
        Prepare ToolBench queries for MCP evaluation.
        This will adapt the StableToolBench dataset format to our evaluation needs.
        """
        
        # Placeholder for when MCP proxy provides the dataset
        # This will load and format the StableToolBench dataset
        
        sample_queries = [
            {
                "query_id": "sample_1",
                "query": "What's the weather like in New York?",
                "expected_tool": "weather_info"
            },
            {
                "query_id": "sample_2", 
                "query": "Count the words in 'Hello world'",
                "expected_tool": "word_count"
            }
        ]
        
        print(f"📋 Prepared {len(sample_queries)} ToolBench queries for MCP evaluation")
        return sample_queries
