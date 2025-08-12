from simplified_toolbench_evaluator import SimplifiedToolBenchEvaluator

def test_simplified_evaluator():
    """Test the simplified ToolBench evaluator."""
    print("=== Testing Simplified ToolBench Evaluator (No Finish Step Validation) ===")
    
    # Test cases with direct, concise responses
    test_cases = [
        {
            "query": "What is the weather in New York?",
            "answer": "The weather in New York is sunny with a high of 75°F and a low of 50°F.",
            "expected_tool": "weather_info",
            "agent_steps": ["weather_info"],
            "available_tools": ["weather_info", "word_count", "reverse_string", "uppercase", "insurance_scorer"]
        },
        {
            "query": "Count the words in 'Hello world'",
            "answer": "There are 2 words in 'Hello world'.",
            "expected_tool": "word_count", 
            "agent_steps": ["word_count"],
            "available_tools": ["weather_info", "word_count", "reverse_string", "uppercase", "insurance_scorer"]
        },
        {
            "query": "Reverse the text 'Python'",
            "answer": "The reversed text is 'nohtyP'.",
            "expected_tool": "reverse_string",
            "agent_steps": ["reverse_string"],
            "available_tools": ["weather_info", "word_count", "reverse_string", "uppercase", "insurance_scorer"]
        },
        {
            "query": "Convert 'hello' to uppercase",
            "answer": "The uppercase version is 'HELLO'.",
            "expected_tool": "uppercase",
            "agent_steps": ["uppercase"],
            "available_tools": ["weather_info", "word_count", "reverse_string", "uppercase", "insurance_scorer"]
        },
        {
            "query": "What's the weather like in London?",
            "answer": "The weather in London is cloudy with a high of 60°F and a low of 45°F.",
            "expected_tool": "weather_info",
            "agent_steps": ["weather_info"],
            "available_tools": ["weather_info", "word_count", "reverse_string", "uppercase", "insurance_scorer"]
        }
    ]
    
    evaluator = SimplifiedToolBenchEvaluator(evaluate_times=1, max_eval_threads=2)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Query: {case['query']}")
        print(f"Answer: {case['answer']}")
        print(f"Expected Tool: {case['expected_tool']}")
        print(f"Agent Steps: {case['agent_steps']}")
        print(f"Available Tools: {case['available_tools']}")
        
        # Test the simplified evaluator
        status, reason = evaluator.check_is_solved(
            query=case["query"],
            expected_tool=case["expected_tool"],
            available_tools=case["available_tools"],
            agent_steps=case["agent_steps"],
            final_answer=case["answer"]
        )
        print(f"Answer Status: {status}")
        print(f"Reason: {reason}")

if __name__ == "__main__":
    test_simplified_evaluator()
