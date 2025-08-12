# LangChain Max Tool Experiment with ToolBench Evaluation

This experiment tests how well LangChain handles increasing numbers of tools using either vllm or local Ollama models by measuring **tool selection accuracy, execution success, and latency plus the StableToolBench solvable pass rate**.

## Overview

The experiment consists of:
1. **MCP Tool Server** (`mcp_tool_server.py`) - Provides tools via MCP protocol
2. **Enhanced Experiment** (`enhanced_maxtool_experiment.py`) - Tests tool selection and execution with ToolBench evaluation
3. **ToolBench Evaluation** -  StableToolBench pass rate evaluation

### Available Tools so far
- `weather_info` - Fetches weather for a location
- `word_count` - Counts words in text
- `reverse_string` - Reverses text
- `uppercase` - Converts text to uppercase
- `insurance_scorer` - Generates random insurance scores

### Test Queries
5 fixed queries, each mapped to a ground truth tool:
1. "What is the weather in New York?" → `weather_info`
2. "How many words are in 'Hello World, this is a test sentence'?" → `word_count`
3. "Reverse this text: Python Experiment" → `reverse_string`
4. "Convert this to uppercase: llamastack" → `uppercase`
5. "Give me an insurance evaluation score" → `insurance_scorer`

## Prerequisites

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 3. Start Ollama and Pull Model
```bash
ollama serve
ollama pull llama3.2:3b-instruct-fp16
```

**Note**: For cluster deployments with vLLM, the model will be automatically loaded by the vLLM server.

## Running the Experiment

### Quick Test (Evaluator Only - ~30 seconds)
```bash
cd src/max_tool_experiment
uv run python test_simplified_evaluator.py
```

### Full Experiment (Complete - ~2-3 minutes)
```bash
cd src/max_tool_experiment
./run_experiment.sh
```

The script will:
- ✅ Check dependencies and start services
- ✅ Run agent execution with ToolBench evaluation and original agent tool call evaluation.
- ✅ Generate comprehensive metrics and reports
- ✅ Display experiment summary
- ✅ Clean up automatically

## Results

### Metrics Measured
- **Agent Metrics**: Tool execution rate, correct tool rate, latency (the same as before)
- **ToolBench Pass Rate**: StableToolBench evaluation (multiple runs for statistical significance)
- **Solve Status**: Solved/Unsolved/Unsure classification (evaluated by gpt model, api key needed.)

### Output Files
- `enhanced_experiment_results.json` - Complete results with both agent and ToolBench metrics

### Sample Results
```json
{
  "agent_metrics": {
    "tool_execution_rate": 100.0,
    "correct_tool_rate": 100.0,
    "average_latency": 5.51
  },
  "toolbench_pass_rate_evaluation": {
    "overall_statistics": {
      "pass_rate": 70.0,
      "std_dev": 10.0
    }
  }
  "detailed_results": [
      {
        "query_id": "query_0",
        "query": "What is the weather in New York?",
        "expected_tool": "weather_info",
        "available_tools": [
          "weather_info",
          "word_count",
          "reverse_string",
          "uppercase",
          "insurance_scorer"
        ],
        "agent_steps": [
          "weather_info"
        ],
        "final_answer": "Current weather conditions in New York are mostly sunny with a high of 75\u00b0F and a low of 50\u00b0F.",
        "is_solved_evaluations": [
          "AnswerStatus.Solved",
          "AnswerStatus.Solved",
          "AnswerStatus.Solved",
          "AnswerStatus.Solved"
        ],
        "pass_rate": 100.0,
        "execution_time": 2.845794200897217
      },
      
        .....
}
```

## Key Files

- `enhanced_maxtool_experiment.py` - Main experiment script
- `simplified_toolbench_evaluator.py` - StableToolBench evaluation
- `mcp_tool_server.py` - MCP tool server
- `run_experiment.sh` - Automated experiment runner
- `test_simplified_evaluator.py` - Evaluator test script

## Customization

### Adding New Tools
1. Add tool implementation to `mcp_tool_server.py`
2. Update `ENHANCED_QUERIES` in `enhanced_maxtool_experiment.py`
or use mcp proxy to link to StableToolBench dataset.

### Modifying Evaluation
Edit `simplified_toolbench_evaluator.py` to customize:
- Judge prompt in `_judge_is_solved()`
- Task solvability check in `_check_task_solvable()`
- Evaluation parameters in `__init__()`
