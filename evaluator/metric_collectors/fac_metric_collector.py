import json
import sys
import os
import requests
from typing import Dict, List, Any
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.utils.module_extractor import register_metric_collector
from evaluator.utils.tool_logger import ToolLogger


@register_metric_collector("fac_metric_collector")
class FACMetricCollector(MetricCollector):
    """
    Final Answer Correctness (FAC) Metric Collector
    
    Collects FAC metrics by directly calling a remote judge model
    with the StableToolBench FAC judge prompt.
    """
    
    FAC_JUDGE_PROMPT = """
Given a query and an answer provided by an AI agent, you now need to determine the answer_status of whether the well solved the query, i.e. whether the need of the query is satisfied. You need to output "Unsolved" or "Solved" and your reason. You must obey the following rules:

You should response "Solved" when:
    1. If the answer well provides the information needed by the query, then it is "Solved". The answer does not need to be perfect, and it only needs to make a genuine attempt to address the query.
	2.	Consider only Completeness:
	    •	The answer attempts to address every part of the query, regardless of whether the information provided is factually correct or accurate, unless there is a severe factual error.
	3.	For Multi-part Queries:
	    •	For queries with multiple parts, all parts must be addressed for the answer to be considered “Solved”.
	4.	Genuine Attempt :
	    •	The answer makes a genuine attempt to provide the requested information or perform the requested task for all parts of the query. This includes scenarios where the answer concludes that “nothing” is a reasonable response (e.g., when the requested information does not exist or is not available, or a possible answer of the query is nothing and the model answers nothing after reasonable attempts).

You should response "Unsolved" when:
    1.	Refusal, Apology, or Non-engagement:
	    •	The answer includes a refusal or apology (e.g., “I'm sorry, I can't help with that”).
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

Here are some examples:

Query 1: I'm planning a movie night with my friends and we want to watch a horror film. Can you recommend some popular horror movies available on streaming platforms in the US? Also, provide the runtime and IMDb ratings for these movies.
Answer 1: Here are some popular horror movies available on streaming platforms in the US:\n\n1. Knives Out\n   - Runtime: 130 minutes\n   - IMDb Rating: 7.9/10\n   - Available on: Netflix, Prime Video, Hulu, Amazon Prime Video\n\n2. Jumanji: The Next Level\n   - Runtime: 110 minutes\n   - IMDb Rating: 6.7/10\n   - Available on: Hulu, Amazon Prime Video, Netflix\n\nPlease note that availability may vary depending on your location and streaming platform subscriptions. Enjoy your movie night!
Answer Status: Solved0304
Reason: The answer addressed all parts of subqueries by providing a list of popular horror movies available on streaming platforms in the US, along with their runtime and IMDb ratings. Whether the film is horror is a factual matter that does not to be checked.

Query 2: I'm a screenwriter looking for inspiration for my next project. Can you help me find a list of critically acclaimed movies available on streaming platforms in the US? Also, provide me with the streaming sources for the movie 'Citizen Kane' and the basic information about the cast, including their names and professions.
Answer 2: The movie 'Citizen Kane' is available on the following streaming platforms:
- HBO Max: [Watch on HBO Max](https://play.hbomax.com/feature/urn:hbo:feature:GXduU_gwwz-NvjAEAAAAC)
- Amazon Prime Video: [Watch on Amazon Prime Video](https://www.amazon.com/Citizen-Kane-Orson-Welles/dp/B000I9YLWG)
- AppleTV+: [Watch on AppleTV+](https://tv.apple.com/us/movie/citizen-kane/umc.cmc.21zj0v11gnqbbqrebieh0vpk7)
- Disney+: Available on Disney+
- Netflix: Available on Netflix
Answer Status: Unsolved
Reason: The answer only addressed the first and second part of the query by providing the streaming sources for the movie 'Citizen Kane' but did not provide information about the cast or a list of critically acclaimed movies available on streaming platforms in the US. The response was incomplete and did not fully address the query.

Query: 
{query}
Answer: 
{answer}

Now give your reason and answer status in the following format:

Answer Status
xxx (can only be "Solved" or "Unsolved")
Reason
xxx
"""
    
    def __init__(self, settings: Dict):
        super().__init__(settings)
        self.logger = ToolLogger("fac_metric_collector.log")
        
        # Metrics storage
        self.query_results = []
        self.total_queries = 0
        self.solved_queries = 0
        self.total_latency = 0.0
        
        # judge model configuration
        self.remote_judge_model = os.getenv('REMOTE_JUDGE_MODEL')
        if not self.remote_judge_model:
            raise ValueError("REMOTE_JUDGE_MODEL environment variable is required")
        
        # Configuration options, show judge model output and detailed explanation
        self.show_judge_output = settings.get('show_judge_output', True)
        self.show_detailed_explanation = settings.get('show_detailed_explanation', True)

    def get_collected_metrics_names(self) -> List[str]:
        return [
            "FAC Solve Rate (%)",
            "FAC Total Queries",
            "FAC Solved Queries", 
            "FAC Unsolved Queries",
            "FAC Average Latency (s)",
            "FAC Total Latency (s)"
        ]

    def set_up(self) -> None:
        """Initialize the FAC metric collector."""
        super().set_up()
        print("🚀 Setting up FAC Metric Collector...")
        print(f"🔍 Using OpenShift URL: {self.remote_judge_model}")
        print("✅ FAC Metric Collector setup complete")

    def prepare_for_measurement(self, query: str) -> None:
        """Prepare for measuring a single query."""
        pass

    def register_measurement(self, query: str, response: Any = None, correct_tool: str = None) -> None:
        """Register measurement for a single query."""
        import time
        
        start_time = time.time()
        
        try:
            # Extract final answer from algorithm response
            final_answer = self.extract_final_answer_from_response(response)
            
            # Evaluate using OpenShift judge model
            evaluation_result = self.evaluate_with_openshift_judge(query, final_answer)
            
            latency = time.time() - start_time
            
            result = {
                "query": query,
                "response": response,
                "final_answer": final_answer,
                "evaluation": evaluation_result["evaluation"],
                "is_solved": evaluation_result["is_solved"],
                "latency": latency,
                "correct_tool": correct_tool
            }
            
            self.total_latency += latency
            self.query_results.append(result)
            
            status_emoji = "✅" if evaluation_result["is_solved"] else "❌"
            print(f"📊 FAC Query Evaluated: {query[:50]}... {status_emoji} (latency: {latency:.2f}s)")
            
            # Show detailed judge model output
            self.log_judge_output(query, final_answer, evaluation_result["evaluation"], evaluation_result["is_solved"])
            
        except Exception as e:
            latency = time.time() - start_time
            result = {
                "query": query,
                "response": response,
                "final_answer": "",
                "evaluation": f"Error: {str(e)}",
                "is_solved": False,
                "latency": latency,
                "error": str(e)
            }
            self.total_latency += latency
            self.query_results.append(result)
            print(f"❌ FAC Query Error: {e}")

    def extract_final_answer_from_response(self, response: Any) -> str:
        """Extract final answer from algorithm response."""
        try:
            # Handle different response types from algorithms
            if response is None:
                return ""
            
            # If it's a string, return it directly
            if isinstance(response, str):
                return response.strip()
            
            # If it's a dict, look for common answer fields
            if isinstance(response, dict):
                # Try common answer field names
                for field in ['answer', 'output', 'result', 'response', 'content']:
                    if field in response:
                        return str(response[field]).strip()
                
                # If no common field, return the whole dict as string
                return str(response)
            
            # For other types, convert to string
            return str(response).strip()
                
        except Exception as e:
            print(f"❌ Error extracting final answer: {e}")
            return str(response) if response else ""

    def evaluate_with_openshift_judge(self, query: str, answer: str) -> Dict[str, Any]:
        """Evaluate query-answer pair using OpenShift judge model."""
        try:
            # Format the prompt
            prompt = self.FAC_JUDGE_PROMPT.format(query=query, answer=answer)
            
            # Prepare payload for OpenShift API
            payload = {
                "prompt": prompt,
                "max_new_tokens": 512,
                "do_sample": False,
                "top_p": 1.0
            }
            
            # Call judge model API
            response = requests.post(
                self.remote_judge_model,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract generated text from response
                generated_text = ""
                if "generated_text" in result:
                    generated_text = result["generated_text"]
                elif "text" in result:
                    generated_text = result["text"]
                elif "response" in result:
                    generated_text = result["response"]
                elif "output" in result:
                    generated_text = result["output"]
                else:
                    generated_text = str(result)
                
                # Extract only the evaluation part (look for the last "Answer Status:" in the response)
                if "Answer Status:" in generated_text:
                    # Find the last occurrence of "Answer Status:" to get the actual evaluation
                    last_evaluation_start = generated_text.rfind("Answer Status:")
                    generated_text = generated_text[last_evaluation_start:]
                
                # Parse the evaluation to determine if solved
                is_solved = self.parse_evaluation_result(generated_text)
                
                return {
                    "evaluation": generated_text,
                    "is_solved": is_solved
                }
            else:
                error_msg = f"API call failed: {response.status_code} - {response.text}"
                print(f"❌ Judge model API error: {error_msg}")
                print(f"🔍 Debug info:")
                print(f"   URL: {self.remote_judge_model}")
                print(f"   Payload: {json.dumps(payload, indent=2)}")
                print(f"   Response: {response.text}")
                return {
                    "evaluation": f"Answer Status: Unsolved\nReason: {error_msg}",
                    "is_solved": False
                }
                
        except Exception as e:
            error_msg = f"Error calling judge model: {e}"
            print(f"❌ {error_msg}")
            print(f"🔍 Debug info:")
            print(f"   URL: {self.remote_judge_model}")
            print(f"   Error: {e}")
            return {
                "evaluation": f"Answer Status: Unsolved\nReason: {error_msg}",
                "is_solved": False
            }

    def parse_evaluation_result(self, evaluation_text: str) -> bool:
        """Parse evaluation text to determine if the query was solved."""
        try:
            # Convert to lowercase for case-insensitive matching
            text = evaluation_text.lower().strip()
            
            # Look for "solved" or "unsolved" in the text
            if "unsolved" in text:
                return False
            elif "solved" in text:
                return True
            else:
                # If unclear, default to unsolved (conservative approach)
                print(f"⚠️ Unclear evaluation result: {evaluation_text[:100]}...")
                return False
                
        except Exception as e:
            print(f"❌ Error parsing evaluation result: {e}")
            return False

    def log_judge_output(self, query: str, answer: str, evaluation: str, is_solved: bool):
        """Log detailed judge model output and explanation."""
        if not self.show_judge_output:
            return
        
        print(f"\n{'='*60}")
        print(f"🔍 JUDGE MODEL EVALUATION")
        print(f"{'='*60}")
        
        print(f"📝 Query: {query}")
        print(f"💬 Answer: {answer}")
        print(f"⚖️  Judge Decision: {'SOLVED' if is_solved else 'UNSOLVED'}")
        
        if self.show_detailed_explanation:
            print(f"\n📋 Full Judge Model Output:")
            print(f"{'-'*40}")
            print(evaluation)
            print(f"{'-'*40}")
        else:
            # Extract just the key parts
            lines = evaluation.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Answer Status') or line.startswith('Reason'):
                    print(f"📋 {line}")
        
        print(f"{'='*60}\n")

    def tear_down(self) -> None:
        """Clean up after all measurements."""
        print("🧹 Starting FAC Metric Collector tear down...")
        
        # Mark collection as complete
        self.collection_active = False
        print(f"🧹 Collection marked as complete: {self.collection_active}")
        
        super().tear_down()
        print("🧹 FAC Metric Collector torn down")

    def report_results(self) -> Dict[str, Any]:
        """Report the collected metrics."""
        print(f"🔍 report_results called, collection_active: {self.collection_active}")
        if self.collection_active:
            raise RuntimeError(f"Metric collector {self.get_name()}: cannot report results while collection is active")
        
        if self.total_queries == 0:
            self.total_queries = len(self.query_results)
        
        if self.total_queries == 0:
            raise RuntimeError("No FAC evaluation results available. Run evaluation first.")
        
        # Count solved queries from our direct evaluations
        solved_queries = sum(1 for result in self.query_results if result.get("is_solved", False))
        solve_rate = (solved_queries / self.total_queries) * 100 if self.total_queries > 0 else 0
        average_latency = self.total_latency / self.total_queries if self.total_queries > 0 else 0
        
        results = {
            "FAC Solve Rate (%)": solve_rate,
            "FAC Total Queries": self.total_queries,
            "FAC Solved Queries": solved_queries,
            "FAC Unsolved Queries": self.total_queries - solved_queries,
            "FAC Average Latency (s)": average_latency,
            "FAC Total Latency (s)": self.total_latency
        }
        
        print(f"📊 FAC Results: {solve_rate:.1f}% solve rate, {average_latency:.2f}s avg latency")
        print(f"📊 Solved: {solved_queries}/{self.total_queries} queries")
        
        return results