import asyncio
import os
from pathlib import Path
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from evaluator.components.mcp_proxy.mcp_proxy import run_mcp_proxy
from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.components.data_provider import get_queries, QuerySpecification, ToolSet, get_tools_from_queries
from evaluator.utils.module_extractor import create_algorithms, create_metric_collectors
from evaluator.eval_spec import EVALUATED_ALGORITHMS, METRIC_COLLECTORS, DATASET_SETTINGS
from evaluator.interfaces.tool_rag_algorithm import ToolRagAlgorithm
from evaluator.utils.csv_logger import CSVLogger
from evaluator.components.llm_provider import get_llm
from dotenv import load_dotenv

load_dotenv()

"""
# To run the experiments, first start the MCP tool server in one terminal:
python evaluator/components/mcp_tool_server.py

# Then run the experiment in another terminal:
python evaluator/run_experiments.py
"""


async def run_all_experiments():

    # Set up the necessary components for the experiments:
    # - the language model
    # - the data to evaluate on
    # - the tools to use
    # - the evaluation metrics to collect and calculate
    # - the algorithms to be evaluated

    print(f"Launching evaluation setup with the following parameters:\n{DATASET_SETTINGS}\n")

    provider_id = os.getenv("LLM_PROVIDER_ID")
    model_id = os.getenv("MODEL_ID")
    base_url = os.getenv("INFERENCE_SERVER_BASE_URL")
    print(f"Connecting to {provider_id} server on {base_url} serving {model_id}...")
    llm = get_llm(provider_id=provider_id, model_id=model_id, base_url=base_url)
    print("Connection established successfully.")

    print("Fetching query dataset...")
    queries = get_queries()
    print(f"Successfully loaded {len(queries)} queries.")

    print("Retrieving available tool definitions...")
    tool_specs = get_tools_from_queries(queries)
    tools = await set_up_mcp(tool_specs)
    print(f"Evaluation will proceed with {len(tools)} tools.")

    print("Loading metric collectors...")
    metric_collectors = create_metric_collectors(METRIC_COLLECTORS)
    print(f"The following metric collectors will be active during evaluation:\n{METRIC_COLLECTORS}\n")

    print("Loading experimental configurations...")
    algorithms_to_compare = create_algorithms(EVALUATED_ALGORITHMS)
    print(f"The following configurations will be executed:\n{EVALUATED_ALGORITHMS}\n")

    # Actually run the experiments
    with CSVLogger(metric_collectors, Path(os.getenv("OUTPUT_PATH")), metadata_columns=["Experiment ID", "Algorithm"]) as logger:
        for i, algo in enumerate(algorithms_to_compare):
            print(f"Running experiment {i+1} of {len(algorithms_to_compare)}: {algo.get_unique_id()}...")
            await run_experiment(algo, llm, tools, queries, metric_collectors)
            logger.log_experiment(meta_values={"Experiment ID": i+1, "Algorithm": algo.get_unique_id()})
            print(f"Experiment {i+1} completed.\n")


async def set_up_mcp(tools_to_provide: ToolSet) -> List[BaseTool]:
    await run_mcp_proxy(tools_to_provide, run_detached=True)
    mcp_proxy_port = os.getenv("MCP_PROXY_LOCAL_PORT", 9000)
    client = MultiServerMCPClient({
        "general": {
            "transport": "streamable_http",
            "url": f"http://127.0.0.1:{mcp_proxy_port}/mcp/"
        }
    })
    return await client.get_tools()


async def run_experiment(algo: ToolRagAlgorithm,
                         llm: BaseChatModel,
                         tools: List[BaseTool],
                         queries: List[QuerySpecification],
                         metric_collectors: List[MetricCollector]
                         ):

    algo.set_up(llm, tools)
    for mc in metric_collectors:
        mc.set_up()

    for i, query_spec in enumerate(queries):
        print(f"Processing query {i+1} of {len(queries)}...")

        for mc in metric_collectors:
            mc.prepare_for_measurement(query_spec)

        response = await algo.process_query(query_spec)

        for mc in metric_collectors:
            mc.register_measurement(query_spec, response=response)

    algo.tear_down()
    for mc in metric_collectors:
        mc.tear_down()


if __name__ == "__main__":
    asyncio.run(run_all_experiments())
