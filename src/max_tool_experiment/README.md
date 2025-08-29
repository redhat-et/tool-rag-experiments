# Max Tool Experiment - StableToolBench Evaluation

This project evaluates LangGraph ReAct agents using StableToolBench metrics with a curated synthetic dataset.

## 🎯 Overview

We use a **manual dataset** of 10 carefully crafted queries to test agent performance:
- **5 simple single-tool queries**: Basic functionality testing
- **5 logical two-tool queries**: Multi-step reasoning testing

## 📁 Project Structure

```
src/max_tool_experiment/
├── synthetic_dataset/                    # Manual dataset
│   ├── test_instruction/
│   │   └── G1_instruction.json          # 10 curated queries
│   └── test_query_ids/
│       └── G1_instruction.json          # Query ID mappings
├── synthetic_evaluation.py              # Main evaluation script
├── step_by_step_evaluation.py          # Pipeline orchestrator
├── run_stabletoolbench_eval.py         # Python-based evaluation runner
├── mcp_tool_server.py                  # MCP tools implementation
├── llm_provider.py                     # LLM configuration
└── synthetic_evaluation_results/       # Evaluation outputs
    ├── raw_answers/
    ├── converted_answers/
    └── evaluation/
```

## 🛠️ Available Tools

The evaluation uses 5 MCP tools:
- `weather_info(loc)`: Get weather for a location
- `word_count(text)`: Count words in text
- `reverse_string(text)`: Reverse a string
- `uppercase(text)`: Convert text to uppercase
- `insurance_scorer()`: Get insurance score

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Load environment variables
export OPENAI_API_KEY="your-key"
export OPENAI_JUDGE_MODEL="gpt-4o-mini"

# Or use .env file
echo "OPENAI_API_KEY=your-key" > .env
echo "OPENAI_JUDGE_MODEL=gpt-4o-mini" >> .env
```

### 2. Run Complete Evaluation
```bash
cd src/max_tool_experiment
uv run python step_by_step_evaluation.py
```

### 3. Run Individual Components

**Agent Evaluation Only:**
```bash
uv run python synthetic_evaluation.py
```

**StableToolBench Evaluation Only:**
```bash
uv run python run_stabletoolbench_eval_python.py
```

## 📊 Evaluation Metrics

The pipeline generates:
- **SoPR (Solvable Pass Rate)**: Percentage of queries successfully solved
- **FAC (Final Answer Correctness)**: Accuracy of final answers
- **Detailed Results**: Query-by-query analysis

## 📝 Dataset Management

### Edit Queries
The dataset is manually maintained in `synthetic_dataset/test_instruction/G1_instruction.json`:

```json
{
  "api_list": [...],
  "query": "Your query here",
  "relevant_APIs": [...],
  "query_id": "synthetic_XXX"
}
```

### Add New Queries
1. Add query to `G1_instruction.json`
2. Update `test_query_ids/G1_instruction.json` with new ID mapping
3. Re-run evaluation

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for agent and evaluation
- `OPENAI_JUDGE_MODEL`: Model for StableToolBench evaluation (default: gpt-4o-mini)

### LLM Configuration
The agent uses OpenAI GPT-3.5-turbo by default. To use different models:
- Edit `synthetic_evaluation.py` → `initialize_llm()` method
- Supported: OpenAI, Ollama

## 📈 Results Interpretation

### SoPR Score
- **High (80%+)**: Agent handles most queries well
- **Medium (40-80%)**: Agent needs improvement
- **Low (<40%)**: Significant issues with tool usage

### Common Issues
- **Incomplete answers**: Agent stops after first tool call
- **Tool response issues**: Empty responses from tools
- **Format problems**: Incorrect final answer structure

## 🛠️ Troubleshooting

### Common Errors
1. **API Key Issues**: Check `OPENAI_API_KEY` in environment
2. **Path Issues**: Ensure working directory is `src/max_tool_experiment`
3. **Tool Import Errors**: Verify `mcp_tool_server.py` exists

### Debug Mode
For detailed debugging, run individual components:
```bash
# Test agent only
uv run python synthetic_evaluation.py

# Test evaluation only  
uv run python run_stabletoolbench_eval_python.py
```

## 📚 Key Files

- **`synthetic_evaluation.py`**: Main agent evaluation with improved ReAct prompt
- **`step_by_step_evaluation.py`**: Complete pipeline orchestrator
- **`run_stabletoolbench_eval.py`**: Python-based StableToolBench runner
- **`mcp_tool_server.py`**: Tool implementations

## 🎯 Design Decisions

1. **Manual Dataset**: Curated queries for better control and testing
2. **Improved ReAct Prompt**: Clear instructions for multi-step execution
3. **Python-based Evaluation**: Reliable environment variable handling
4. **Component-based Architecture**: Easy to debug and modify individual parts

## 📄 License

See LICENSE file for details.
