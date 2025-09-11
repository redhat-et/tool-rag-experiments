import json
import os
from typing import Tuple, List
from pathlib import Path


def get_data() -> List[Tuple[str, str]]:
    """Load queries from the synthetic dataset."""
    try:
        # Get dataset path from environment variable
        dataset_path = Path(os.getenv("INSTRUCTION_DATASET_PATH"))
        
        if not dataset_path.exists():
            print(f"⚠️ Dataset not found at {dataset_path}")
            print("Using fallback queries")
            return get_fallback_queries()
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        queries = []
        for item in data:
            query = item.get("query", "")
            # Extract the primary tool from the API list
            api_list = item.get("api_list", [])
            if api_list:
                primary_tool = api_list[0].get("api_name", "unknown")
            else:
                primary_tool = "unknown"
            
            queries.append((query, primary_tool))
        
        print(f"✅ Loaded {len(queries)} queries from synthetic dataset")
        return queries
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Using fallback queries")
        return get_fallback_queries()


def get_fallback_queries() -> List[Tuple[str, str]]:
    """Fallback queries if dataset loading fails."""
    return [
        ("What is the weather in New York?", "weather_info"),
        ("How many words are in 'Hello World, this is a test sentence'?", "word_count"),
        ("Reverse this text: Python Experiment", "reverse_string"),
        ("Convert this to uppercase: llamastack", "uppercase"),
        ("Give me an insurance evaluation score", "insurance_scorer")
    ]
