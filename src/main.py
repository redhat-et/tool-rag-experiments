import os
from llama_stack_client import LlamaStackClient
from src.tools.info import tool_use_cases
from src.utils.env import Env
from src.utils.data_loader import load_queries
from src.utils.logging_config import setup_logger
from src.tools.metrics import get_analysis_plots
import test_runner

def insert_tool_embedding(tool_name, tool_use_case, vector_db_id: str, client: LlamaStackClient):
    chunk = {
        "content": tool_use_case,
        "mime_type": "text/plain",
        "metadata": {
            "tool_name": tool_name,
            "document_id": tool_name
        }
    }

    client.vector_io.insert(vector_db_id=vector_db_id, chunks=[chunk])

def fill_vector_db(vector_db_id: str, client: LlamaStackClient, queries):
    query_set = set()

    for query in queries:
        tool_name = query["tool_call"]

        if tool_name not in query_set:
            insert_tool_embedding(
                tool_name=tool_name,
                tool_use_case=tool_use_cases[tool_name],
                vector_db_id=vector_db_id,
                client=client
            )
            query_set.add(tool_name)


def main():
    logger = setup_logger()
    client_tool_queries_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "../eval/", "client_tool_queries.json")
    results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "../results/", "client_tool_metrics.csv")

    base_url = Env()['REMOTE_BASE_URL']
    if not base_url:
        logger.error("REMOTE_BASE_URL environment variable not set")
        return

    llama_client = LlamaStackClient(base_url=base_url)
    models = ["meta-llama/Llama-3.2-3B-Instruct", ]
    vector_db_id = "tool_use_case_test_vdb"
    emodels = llama_client.models.list()
    embedding_model = (
        em := next(m for m in emodels if m.model_type == "embedding")
    ).identifier
    embedding_dimension = 384

    _ = llama_client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        provider_id="faiss",
    )

    queries = load_queries(client_tool_queries_path)

    # fill_vector_db(vector_db_id, llama_client, queries)

    total_tests = 0
    successful_tests = 0

    for model in models:
        logger.info(f"\n=== Testing with model: {model} ===\n")

        if not queries:
            logger.info(f"No queries found")
            continue

        for query_obj in queries:
            total_tests += 1
            success = test_runner.run_client_tool_test(
                model,
                vector_db_id,
                query_obj,
                llama_client,
                logger
            )
            if success:
                successful_tests += 1

    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Successful tests: {successful_tests}")
    logger.info(f"Failed tests: {total_tests - successful_tests}")
    if total_tests > 0:
        success_rate = (successful_tests / total_tests) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")

    logger.info(f"\n=== Generating plots ===")
    get_analysis_plots(results_path)

if __name__ == "__main__":
    main()
