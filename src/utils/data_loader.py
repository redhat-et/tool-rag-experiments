import json
from typing import List, Dict

def load_queries(file_path: str) -> List[Dict[str, str]]:
    """Load query strings from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        if not isinstance(data, dict) or 'queries' not in data:
            raise ValueError(f"Invalid JSON format in {file_path}")

        return data['queries']
    except FileNotFoundError:
        print(f"Query file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return []