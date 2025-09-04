# Max Tool Experiment - FAC-Only Evaluation

This project evaluates LangGraph ReAct agents using **Final Answer Correctness (FAC)** metric with a curated synthetic dataset.

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
├── fac_only_evaluation.py               # Main FAC evaluation script
├── stabletoolbench_fac_eval.py          # FAC evaluation runner (modified original StableToolBench)
├── mcp_tool_server.py                   # MCP tools implementation
├── llm_provider.py                      # LLM configuration
└── fac_evaluation_results/              # FAC evaluation outputs
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
Edit the `.env` file using `.env.example` as a reference.

### 2. Run FAC Evaluation
```bash
./run_experiment.sh
```


## 📊 Evaluation Metrics

The pipeline generates:
- **FAC (Final Answer Correctness)**: Accuracy of final answers using Openshift hosted LLM judge
- **Detailed Results**: Query-by-query analysis with reasoning
- **Performance Metrics**: Success rate and latency

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
- `OPENAI_API_KEY`: OpenAI API key for agent
- `OPENAI_MODEL`: Model for the agent (default: gpt-4o-mini)
- `USE_OLLAMA`: Set to 'true' to use Ollama instead of OpenAI
- `OLLAMA_MODEL`: Ollama model name (default: llama3.2:3b)
- `OPENSHIFT_EVALUATOR_URL`: OpenShift evaluator API URL for FAC evaluation

### LLM Configuration
The system supports both OpenAI and Ollama models:
- **OpenAI**: Fast, reliable, requires API key
- **Ollama**: Local, free, may be slower

## 🔍 FAC Evaluation Process

1. **Agent Execution**: LangGraph ReAct agent processes each query
2. **Answer Extraction**: Final answers are extracted from agent responses
3. **FAC Evaluation**: OpenShift hosted model evaluates answer correctness
4. **Results Storage**: Detailed results saved to `fac_evaluation_results/`

## 🌐 OpenShift Configuration

### FAC Evaluation Model
The FAC evaluation uses your OpenShift hosted model instead of local vLLM:

### Get Your OpenShift Route
```bash
# Get your OpenShift route
ROUTE=$(oc get route evaluator-api -o jsonpath='{.spec.host}')
echo "Your evaluator API URL: https://$ROUTE"

# Test the API
curl -s https://$ROUTE/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello in one sentence.","max_new_tokens":32,"do_sample":false,"top_p":1.0}' | jq .
```

## 📈 Understanding Results

### FAC Results
- **Solved**: Answer addresses all parts of the query
- **Unsolved**: Answer incomplete or doesn't address the query
- **Reason**: Detailed explanation of the evaluation

### Success Rate
- Percentage of queries successfully solved
- Based on FAC evaluation, not just tool execution

## 🚨 Troubleshooting

### Common Issues
1. **Missing API Key**: Ensure `OPENAI_API_KEY` is set
2. **Model Not Found**: Check `OPENAI_JUDGE_MODEL` is valid
3. **Tool Import Errors**: Ensure MCP server is running

### Debug Mode
Run individual components to isolate issues:
```bash
# Test MCP tools only
uv run python mcp_tool_server.py

# Test StableToolBench FAC evaluation only
uv run python stabletoolbench_fac_eval.py
```



## 📚 Technical Details

### LangGraph ReAct Agent
- Uses structured reasoning with Thought/Action/Observation format
- Integrates directly with MCP tools
- Generates final answers in JSON format

### FAC Evaluation
- Uses modified original StableToolBench code with OpenShift support
- Evaluates completeness, not factual accuracy
- Provides detailed reasoning for each evaluation
- Supports both local vLLM and OpenShift hosted models

### Data Flow
```
Synthetic Dataset → LangGraph Agent → Answer Extraction → FAC Evaluation → Results
```