from typing import List


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
