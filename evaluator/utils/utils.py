import re
from typing import Any

from evaluator.eval_spec import VERBOSE


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)


def print_verbose(message):
    if VERBOSE:
        print(message)


def print_iterable_verbose(label, iterable):
    print_verbose(label)
    for item in iterable:
        print_verbose(item)


def extract_final_answer_from_response(response: Any) -> str:
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
            if 'messages' in response:
                return response["messages"][-1].content

            # If no common field, return the whole dict as string
            return str(response)

        # For other types, convert to string
        return str(response).strip()

    except Exception as e:
        print(f"❌ Error extracting final answer: {e}")
        print(f"🔍 Faulty response: {response}")
        return str(response) if response else ""


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from free text."""
    if not text:
        return text
    out = _THINK_BLOCK.sub("", text)  # remove well-formed blocks
    out = _UNCLOSED_THINK.sub("", out)  # guard: remove tail after an unclosed <think>
    # collapse excessive blank lines/spaces created by removals
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out
