"""
StableToolBench Format Sample Data

Pure data file containing the 5 MCP server tools in StableToolBench format.
Used for tool sequence validation experiments.
"""

# Sample data matching StableToolBench format - using the actual 5 tools
SAMPLE_STABLETB_DATA = [
    {
        "api_list": [
            {
                "category_name": "Weather",
                "tool_name": "MCPToolServer",
                "api_name": "weather_info",
                "api_description": "Fetches weather information for a specified location",
                "required_parameters": [
                    {
                        "name": "location", 
                        "type": "string",
                        "description": "The city and state/country for weather lookup",
                        "default": "New York"
                    }
                ],
                "optional_parameters": [],
                "method": "GET",
                "template_response": {
                    "temperature": "str",
                    "condition": "str",
                    "location": "str"
                }
            }
        ],
        "query": "What is the weather in New York?",
        "relevant APIs": [
            ["MCPToolServer", "weather_info"]
        ],
        "query_id": 1001
    },
    {
        "api_list": [
            {
                "category_name": "Text Processing",
                "tool_name": "MCPToolServer",
                "api_name": "word_count",
                "api_description": "Counts the number of words in the provided text",
                "required_parameters": [
                    {
                        "name": "text",
                        "type": "string", 
                        "description": "The text to count words in",
                        "default": "Hello World"
                    }
                ],
                "optional_parameters": [],
                "method": "POST",
                "template_response": {
                    "word_count": "int",
                    "text": "str"
                }
            }
        ],
        "query": "How many words are in 'Hello World, this is a test sentence'?",
        "relevant APIs": [
            ["MCPToolServer", "word_count"]
        ],
        "query_id": 1002
    },
    {
        "api_list": [
            {
                "category_name": "Text Processing",
                "tool_name": "MCPToolServer",
                "api_name": "reverse_string",
                "api_description": "Reverses the provided text string",
                "required_parameters": [
                    {
                        "name": "text",
                        "type": "string",
                        "description": "The text to reverse",
                        "default": "Hello"
                    }
                ],
                "optional_parameters": [],
                "method": "POST",
                "template_response": {
                    "reversed_text": "str",
                    "original_text": "str"
                }
            }
        ],
        "query": "Reverse this text: Python Experiment",
        "relevant APIs": [
            ["MCPToolServer", "reverse_string"]
        ],
        "query_id": 1003
    },
    {
        "api_list": [
            {
                "category_name": "Text Processing",
                "tool_name": "MCPToolServer",
                "api_name": "uppercase",
                "api_description": "Converts text to uppercase",
                "required_parameters": [
                    {
                        "name": "text",
                        "type": "string",
                        "description": "The text to convert to uppercase",
                        "default": "hello"
                    }
                ],
                "optional_parameters": [],
                "method": "POST",
                "template_response": {
                    "uppercase_text": "str",
                    "original_text": "str"
                }
            }
        ],
        "query": "Convert this to uppercase: llamastack",
        "relevant APIs": [
            ["MCPToolServer", "uppercase"]
        ],
        "query_id": 1004
    },
    {
        "api_list": [
            {
                "category_name": "Insurance",
                "tool_name": "MCPToolServer",
                "api_name": "insurance_scorer",
                "api_description": "Generates a random insurance evaluation score",
                "required_parameters": [
                    {
                        "name": "age",
                        "type": "integer",
                        "description": "The age of the person for insurance evaluation",
                        "default": "30"
                    },
                    {
                        "name": "risk_factors",
                        "type": "string",
                        "description": "Comma-separated risk factors",
                        "default": "none"
                    }
                ],
                "optional_parameters": [],
                "method": "POST",
                "template_response": {
                    "score": "float",
                    "factors_considered": "list",
                    "recommendation": "str"
                }
            }
        ],
        "query": "Give me an insurance evaluation score",
        "relevant APIs": [
            ["MCPToolServer", "insurance_scorer"]
        ],
        "query_id": 1005
    }
]

def get_sample_data():
    """Get the sample StableToolBench format data."""
    return SAMPLE_STABLETB_DATA

def extract_golden_tools(stabletb_item: dict) -> list[str]:
    """Extract golden tools from StableToolBench item's 'relevant APIs' field."""
    relevant_apis = stabletb_item.get("relevant APIs", [])
    golden_tools = []
    for api_info in relevant_apis:
        if len(api_info) >= 2:
            # Use api_name as the tool identifier
            golden_tools.append(api_info[1])
    return golden_tools
