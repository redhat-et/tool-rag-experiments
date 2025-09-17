import functools
import os
from typing import List

from evaluator.utils.utils import print_verbose


class ToolLogger(object):
    def __init__(self, log_file: str):
        self.log_file = log_file

    def get_executed_tools(self) -> List[str]:
        """Read tool names from the log file."""
        executed_tools = []
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    if line.strip().startswith("[TOOL]"):
                        tool_name = line.strip().split("[TOOL] ")[1]
                        executed_tools.append(tool_name)
        except FileNotFoundError:
            pass
        self.clear_log()
        return executed_tools

    def clear_log(self):
        """Clear the log file."""
        try:
            with open(self.log_file, "w") as f:
                f.write("")
        except Exception as e:
            print(f"❌ Error clearing tool_log.txt: {e}")


def log_tool(tool_name):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create file if it doesn't exist
            with open(os.getenv("TOOL_LOG_PATH"), "a") as f:
                f.write(f"[TOOL] {tool_name}\n")
            print_verbose(f"Executing tool {tool_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
