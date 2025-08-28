import os

from mcp.server.fastmcp import FastMCP
import random
import functools
from dotenv import load_dotenv

load_dotenv()


def log_tool(tool_name):
    print(f"Logging tool: {tool_name}")
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create file if it doesn't exist
            with open(os.getenv("TOOL_LOG_PATH"), "a") as f:
                f.write(f"[TOOL] {tool_name}\n")
            return func(*args, **kwargs)
        return wrapper
    return decorator


mcp = FastMCP("General")


@mcp.tool()
@log_tool("weather_info")
def weather_info(loc: str) -> str:
    """Fetches the current weather for a given location."""
    return f" Weather in {loc} is sunny."

@mcp.tool()
@log_tool("word_count")
def word_count(text: str) -> str:
    """Counts the number of words in the given text."""
    return f"Word count: {len(text.split())}"

@mcp.tool()
@log_tool("reverse_string")
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return f"Reversed text: {text[::-1]}"

@mcp.tool()
@log_tool("uppercase")
def uppercase(text: str) -> str:
    """Converts the given string to uppercase."""
    return f"Uppercase text: {text.upper()}"

@mcp.tool()
@log_tool("insurance_scorer")
def insurance_scorer() -> str:
    """Generates a random insurance score between 1 and 100."""
    return f"Insurance score: {random.randint(1, 100)}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
