#!/usr/bin/env python3
"""
Custom FAC (Final Answer Correctness) evaluation using OpenShift vLLM service
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

FAC_PROMPT = """
Given a query and an answer provided by an AI agent, you now need to determine the answer_status of whether the agent well solved the query, i.e. whether the need of the query is satisfied. You need to output "Unsolved" or "Solved" and your reason. You must obey the following rules:

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
    1. DO NOT evaluate factual accuracy or correctness of the information provided based on your knowledge. Assume that the information provided is accurate and focus solely on whether the answer attempts to address all parts of the query, unless there is a severe factual error that conflicts common knowledge.
	2.	Focus on Final Answer: Only the final answer is provided and should be considered, disregarding any processes that were used to generate the answer. You only need to judge whether the information need is satisfied.
	3.	Answer Completion: The agent does not need to detail how it arrived at the answer, only that the answer itself is complete and attempts to address the query.

Query: {query}
Answer: {answer}

Now give your reason and answer status in the following format:

Answer Status: [Solved/Unsolved]
Reason: [Your detailed reason]
"""

def call_openai_service(prompt: str) -> str:
    """Call OpenAI API with the given prompt."""
    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_JUDGE_MODEL', 'gpt-4.1-nano'),
            messages=[
                {"role": "system", "content": "You are an expert evaluator for AI agent responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"❌ Error calling OpenAI service: {e}")
        return ""

def evaluate_answer(query: str, answer: str) -> Dict[str, Any]:
    """Evaluate a single answer using the vLLM service."""
    prompt = FAC_PROMPT.format(query=query, answer=answer)
    
    print(f"🔍 Evaluating query: {query[:50]}...")
    
    response = call_openai_service(prompt)
    
    # Parse the response to extract status and reason
    status = "Unsolved"  # Default
    reason = "Failed to parse evaluation response"
    
    if response:
        # Look for "Answer Status:" in the response
        if "Answer Status:" in response:
            status_line = response.split("Answer Status:")[1].split("\n")[0].strip()
            if "Solved" in status_line:
                status = "Solved"
            elif "Unsolved" in status_line:
                status = "Unsolved"
        
        # Look for "Reason:" in the response
        if "Reason:" in response:
            reason = response.split("Reason:")[1].strip()
    
    return {
        "status": status,
        "reason": reason,
        "raw_response": response
    }

def run_fac_evaluation(converted_answer_path: str, save_path: str, test_ids_path: str, test_set: str = "G1_instruction"):
    """Run FAC evaluation on the converted answers."""
    
    # Load test IDs
    test_ids_file = Path(test_ids_path) / f"{test_set}.json"
    with open(test_ids_file, 'r') as f:
        test_ids = json.load(f)
    
    # Load converted answers
    converted_file = Path(converted_answer_path) / "langgraph_react" / f"{test_set}.json"
    with open(converted_file, 'r') as f:
        converted_answers = json.load(f)
    
    # Create save directory
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each answer
    results = {}
    total_queries = len(test_ids)
    solved_count = 0
    
    print(f"🚀 Starting FAC evaluation for {total_queries} queries...")
    
    for query_id in test_ids:
        if query_id in converted_answers:
            query = converted_answers[query_id]["query"]
            # Extract final answer from the nested structure
            answer_data = converted_answers[query_id]["answer"]["final_answer"]
            # Parse the JSON string to get the actual final answer
            try:
                answer_json = json.loads(answer_data)
                answer = answer_json.get("final_answer", answer_data)
            except json.JSONDecodeError:
                answer = answer_data
            
            evaluation = evaluate_answer(query, answer)
            results[query_id] = evaluation
            
            if evaluation["status"] == "Solved":
                solved_count += 1
            
            print(f"  {query_id}: {evaluation['status']}")
        else:
            print(f"  {query_id}: Not found in converted answers")
            results[query_id] = {
                "status": "Unsolved",
                "reason": "Query not found in converted answers",
                "raw_response": ""
            }
    
    # Calculate solve rate
    solve_rate = (solved_count / total_queries) * 100 if total_queries > 0 else 0
    
    # Save results
    results_file = save_dir / f"{test_set}_langgraph_react_fac.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary = {
        "test_set": test_set,
        "model": "langgraph_react",
        "total_queries": total_queries,
        "solved_count": solved_count,
        "solve_rate": solve_rate,
        "results": results
    }
    
    summary_file = save_dir / f"{test_set}_langgraph_react_fac_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ FAC evaluation completed!")
    print(f"📊 Results saved to: {results_file}")
    print(f"📊 Summary saved to: {summary_file}")
    print(f"🎯 Solve rate: {solve_rate:.1f}% ({solved_count}/{total_queries})")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Custom FAC evaluation using OpenShift vLLM")
    parser.add_argument("--converted_answer_path", type=str, required=True, help="Path to converted answers")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--test_ids", type=str, required=True, help="Path to test IDs")
    parser.add_argument("--test_set", type=str, default="G1_instruction", help="Test set name")
    
    args = parser.parse_args()
    
    success = run_fac_evaluation(
        args.converted_answer_path,
        args.save_path,
        args.test_ids,
        args.test_set
    )
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
