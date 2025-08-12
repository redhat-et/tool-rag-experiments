#!/usr/bin/env python3

import json
import os
import statistics
import random
import backoff
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

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

class SimplifiedToolBenchEvaluator:
    """
    Simplified ToolBench-style evaluator that removes Finish step validation
    while keeping all other StableToolBench functionality.
    """
    
    def __init__(self, evaluate_times: int = 4, judge_model: str = None, max_eval_threads: int = 4):
        self.evaluate_times = evaluate_times
        self.max_eval_threads = max_eval_threads
        self.results: List[PassRateResult] = []

        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Please add it to your .env or environment.")

        self.judge_model = judge_model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-nano")

    def _check_task_solvable(self, query: str, available_tools: List[str]) -> str:
        """Check if a task is solvable with the available tools using StableToolBench's logic."""
        
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
            print(f"⚠️ Task solvability check error: {type(e).__name__}: {e}")
            status = "Unsure"

        return status

    @backoff.on_exception(backoff.expo, Exception, max_time=15)
    def _judge_is_solved(self, query: str, answer: str) -> AnswerStatus:
        """Call OpenAI using StableToolBench's exact check_answer_status function."""
        
        # StableToolBench's exact prompt template from fac_eval.py
        function_description = {
            "name": "check_answer_status",
            "description": """Given a query and an answer provided by an AI agent, you now need to determine the answer_status of whether the well solved the query, i.e. whether the need of the query is satisfied. You need to output "Unsolved" or "Solved" and your reason. You must obey the following rules:

You should response "Solved" when:
    1. If the answer well provides the information needed by the query, then it is "Solved". The answer does not need to be perfect, and it only needs to make a genuine attempt to address the query.
	2.	Consider only Completeness:
	    •	The answer attempts to address every part of the query, regardless of whether the information provided is factually correct or accurate, unless there is a severe factual error.
	3.	For Multi-part Queries:
	    •	For queries with multiple parts, all parts must be addressed for the answer to be considered "Solved".
	4.	Genuine Attempt :
	    •	The answer makes a genuine attempt to provide the requested information or perform the requested task for all parts of the query. This includes scenarios where the answer concludes that "nothing" is a reasonable response (e.g., when the requested information does not exist or is not available, or a possible answer of the query is nothing and the model answers nothing after reasonable attempts).

You should response "Unsolved" when:
    1.	Refusal, Apology, or Non-engagement:
	    •	The answer includes a refusal or apology (e.g., "I'm sorry, I can't help with that").
	    •	The answer does not directly engage with or address the query in any way.
	2.	Multi-part Queries:
	    •	If the query has multiple parts and at least one part is not well addressed.
    3. Severe Factual Error:
        •   If the answer contains a severe factual error that significantly impacts the usefulness of the information provided.

Additional Guidelines:
    1. VERY IMPORTANT: DO NOT BE TOO HARSH. The model does not need to be perfect, and the answer does not need to be flawless. It only needs to make a genuine attempt to address the query.
    1. DO NOT evaluate factual accuracy or correctness of the information provided based on your knowledge. Assume that the information provided is accurate and focus solely on whether the answer attempts to address all parts of the query, unless there is a severe factual error that conficts common knowledge.
	2.	Focus on Final Answer: Only the final answer is provided and should be considered, disregarding any processes that were used to generate the answer. You only need to judge whether the information need is satisfied.
	3.	Answer Completion: The agent does not need to detail how it arrived at the answer, only that the answer itself is complete and attempts to address the query.

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
            
            function_args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
            verdict = function_args["answer_status"]
            
            # Convert to AnswerStatus enum
            if verdict == "Solved":
                return AnswerStatus.SOLVED
            elif verdict == "Unsure":
                return AnswerStatus.UNSURE
            else:
                return AnswerStatus.UNSOLVED
            
        except Exception as e:
            print(f"⚠️ Judge error: {type(e).__name__}: {e}")
            return AnswerStatus.UNSOLVED

    def check_is_solved(self, query: str, expected_tool: str, available_tools: List[str], 
                       agent_steps: List[str], final_answer: str) -> Tuple[AnswerStatus, str]:
        """
        Core evaluation function that incorporates StableToolBench logic WITHOUT Finish step validation.
        """
        
        # Empty answer ⇒ Unsolved
        if not final_answer or final_answer.strip() == "":
            return AnswerStatus.UNSOLVED, "No final answer provided"

        # Expected tool must appear in the execution steps
        if expected_tool not in " ".join(agent_steps):
            return AnswerStatus.UNSOLVED, "Expected tool was not used"

        # Check task solvability first (StableToolBench step)
        task_status = self._check_task_solvable(query, available_tools)
        if task_status == "Unsolvable":
            return AnswerStatus.UNSOLVED, f"Task marked as unsolvable: {task_status}"

        # Delegate judgement to OpenAI evaluator (StableToolBench's core evaluation)
        status = self._judge_is_solved(query, final_answer)
        if status == AnswerStatus.SOLVED:
            return AnswerStatus.SOLVED, "OpenAI judge marked as Solved"
        elif status == AnswerStatus.UNSURE:
            return AnswerStatus.UNSURE, "OpenAI judge marked as Unsure"
        else:
            return AnswerStatus.UNSOLVED, "OpenAI judge marked as Unsolved"

    def evaluate_query_pass_rate(self, query_id: str, query: str, expected_tool: str,
                                available_tools: List[str], agent_steps: List[str], 
                                final_answer: str, execution_time: float) -> PassRateResult:
        """
        Evaluate a single query multiple times to calculate pass rate.
        Following StableToolBench's evaluation methodology with multi-threading.
        """
        
        evaluations = []
        
        # Run evaluation multiple times for statistical significance
        for i in range(self.evaluate_times):
            is_solved, reason = self.check_is_solved(
                query, expected_tool, available_tools, agent_steps, final_answer
            )
            evaluations.append(is_solved)
        
        # Calculate pass rate using StableToolBench's scoring method
        solved_count = sum(1 for status in evaluations if status == AnswerStatus.SOLVED)
        unsure_count = sum(1 for status in evaluations if status == AnswerStatus.UNSURE)
        
        # StableToolBench scoring: Solved = 1.0, Unsure = 0.5, Unsolved = 0.0
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

    def evaluate_batch_with_threading(self, queries_data: List[Dict[str, Any]]) -> List[PassRateResult]:
        """
        Evaluate multiple queries using ThreadPoolExecutor like StableToolBench.
        """
        
        def evaluate_single_query(query_data):
            return self.evaluate_query_pass_rate(**query_data)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_eval_threads) as executor:
            # Submit all evaluation tasks
            future_to_query = {
                executor.submit(evaluate_single_query, query_data): query_data 
                for query_data in queries_data
            }
            
            # Process completed evaluations with progress bar
            for future in tqdm(as_completed(future_to_query), total=len(queries_data), ncols=100):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error evaluating query: {e}")
        
        return results

    def calculate_overall_pass_rate(self) -> Dict[str, float]:
        """
        Calculate overall pass rate statistics following StableToolBench methodology.
        """
        if not self.results:
            return {"pass_rate": 0.0, "std_dev": 0.0, "total_queries": 0}
        
        # Calculate pass rates for each evaluation run (following StableToolBench logic)
        pass_rates_by_run = []
        for run_idx in range(self.evaluate_times):
            score = 0
            for result in self.results:
                status = result.is_solved_evaluations[run_idx]
                if status == AnswerStatus.SOLVED:
                    score += 1.0
                elif status == AnswerStatus.UNSURE:
                    score += 0.5
            
            pass_rates_by_run.append(score / len(self.results))
        
        # Calculate statistics using StableToolBench's method
        overall_pass_rate = sum(pass_rates_by_run) / len(pass_rates_by_run)
        std_dev = np.std(pass_rates_by_run).item() if len(pass_rates_by_run) > 1 else 0.0
        
        return {
            "pass_rate": overall_pass_rate * 100,  # Convert to percentage
            "std_dev": std_dev * 100,  # Convert to percentage
            "total_queries": len(self.results),
            "pass_rates_by_run": [rate * 100 for rate in pass_rates_by_run]
        }

    def generate_toolbench_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report in StableToolBench format."""
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
                "max_eval_threads": self.max_eval_threads,
                "judge_model": self.judge_model,
                "total_queries_evaluated": len(self.results)
            }
        }

    def save_results(self, filename: str):
        """Save results in StableToolBench-compatible format."""
        report = self.generate_toolbench_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print summary statistics following StableToolBench format."""
        stats = self.calculate_overall_pass_rate()
        
        print(f"\n=== Simplified ToolBench Pass Rate Evaluation Results ===")
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Pass Rate: {stats['pass_rate']:.1f}%")
        print(f"Standard Deviation: {stats['std_dev']:.1f}%")
        
        # Print per-run results
        print(f"\nPass Rates by Evaluation Run:")
        for i, rate in enumerate(stats['pass_rates_by_run']):
            print(f"  Run {i+1}: {rate:.1f}%")
        
        print(f"\nEvaluation Configuration:")
        print(f"  Evaluate Times: {self.evaluate_times}")
        print(f"  Max Threads: {self.max_eval_threads}")
        print(f"  Judge Model: {self.judge_model}")
        print(f"  Note: Finish step validation removed for compatibility with current agent format")
