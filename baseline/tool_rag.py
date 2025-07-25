import os
import requests
from typing import List
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

from langchain_community.vectorstores import Milvus, MilvusLite, VectorStore
from pymilvus import connections, utility

# Configuration parameters
MCP_SERVERS = [
    "http://mcp-server-1.local"
]

COLLECTION_NAME = "tools_collection"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_LITE_DIR = "./milvus_lite_data"
USE_MILVUS_LITE = os.environ.get("USE_MILVUS_LITE", "true").lower() == "true"  # Set to "false" to use server

VLLM_ENDPOINT_URL = "http://localhost:8000/v1"
MODEL_ID = "granite32-8b"

TOOL_SELECTION_K = 3

TEST_PROMPT = "Can you generate a number between 1 and 42?"


def fetch_tool_definitions() -> List[dict]:
    tool_defs = []
    for server in MCP_SERVERS:
        try:
            response = requests.get(f"{server}/tools")
            response.raise_for_status()
            tools = response.json()
            for tool in tools:
                if "name" in tool and "description" in tool:
                    tool_defs.append({
                        "name": tool["name"],
                        "description": tool["description"]
                    })
        except Exception as e:
            print(f"[WARN] Failed to fetch tools from {server}: {e}")
    return tool_defs


def connect_to_tool_db(embeddings: HuggingFaceEmbeddings, connection_args: dict) -> VectorStore:
    print(f"[INFO] Creating new Milvus Lite collection.")
    tool_defs = fetch_tool_definitions()
    if not tool_defs:
        raise RuntimeError("No tools retrieved from MCP servers.")
    docs = [Document(page_content=tool["description"], metadata={"name": tool["name"]}) for tool in tool_defs]
    if "uri" in connection_args:
        return MilvusLite.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args=connection_args
        )
    return Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args=connection_args
    )


def get_or_index_tools():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    connection_args = {"uri": MILVUS_LITE_DIR} if USE_MILVUS_LITE else {"host": MILVUS_HOST, "port": MILVUS_PORT}

    if USE_MILVUS_LITE:
        path = os.path.join(MILVUS_LITE_DIR, COLLECTION_NAME)
        if os.path.exists(path):
            print(f"[INFO] Loading Milvus Lite collection from: {path}")
            return MilvusLite(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args=connection_args
            )

        return connect_to_tool_db(embeddings, connection_args)

    # USE_MILVUS_LITE == False
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        print(f"[INFO] Loading Milvus server collection: {COLLECTION_NAME}")
        return Milvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args=connection_args
        )

    return connect_to_tool_db(embeddings, connection_args)


def construct_prompt(tools: List[Document], user_prompt: str) -> str:
    tool_text = "\n".join([f"{doc.metadata['name']}: {doc.page_content}" for doc in tools])
    return f"""You are an expert AI assistant.

The following are relevant tool definitions:
{tool_text}

Based on the above tools, respond to the following user request:
{user_prompt}
"""


def run_llm(augmented_prompt: str) -> str:
    llm = OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}"))
    return chain.run(input=augmented_prompt)


def tool_rag_pipeline(user_prompt: str) -> str:
    vectorstore = get_or_index_tools()
    relevant_tools = vectorstore.similarity_search(user_prompt, k=TOOL_SELECTION_K)
    extended_prompt = construct_prompt(relevant_tools, user_prompt)

    llm = ChatOpenAI(base_url=VLLM_ENDPOINT_URL, api_key="fake-key", model=MODEL_ID, temperature=0)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}"))
    return chain.run(input=extended_prompt)


if __name__ == "__main__":
    prompt = TEST_PROMPT
    print("[INFO] Using Milvus Lite" if USE_MILVUS_LITE else "[INFO] Using Milvus server")
    answer = tool_rag_pipeline(prompt)
    print("=== RESPONSE ===")
    print(answer)
