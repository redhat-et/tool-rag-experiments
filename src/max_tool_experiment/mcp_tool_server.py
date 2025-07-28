from mcp.server.fastmcp import FastMCP
import random

mcp = FastMCP("General")

@mcp.tool()
def weather_info(loc: str) -> str:
    """Fetches the current weather for a given location."""
    return f"Weather in {loc} is sunny."

@mcp.tool()
def word_count(text: str) -> str:
    """Counts the number of words in the given text."""
    return f"Word count: {len(text.split())}"

@mcp.tool()
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return f"Reversed text: {text[::-1]}"

@mcp.tool()
def uppercase(text: str) -> str:
    """Converts the given string to uppercase."""
    return f"Uppercase text: {text.upper()}"

@mcp.tool()
def insurance_scorer() -> str:
    """Generates a random insurance score between 1 and 100."""
    return f"Insurance score: {random.randint(1, 100)}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http") 