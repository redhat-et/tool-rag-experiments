import json
import os
from evaluator.algorithms.basic_tool_rag_algorithm import BasicToolRagAlgorithm
from evaluator.algorithms.model.e5 import E5Model
from evaluator.components.data_provider import QuerySpecification
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from evaluator.components.llm_provider import get_llm

@tool
def java_code_compiler(query: str):
    """A dummy tool for Java Code Compiler."""
    return f"Received query: {query}"

class E5EmbeddingsWrapper(Embeddings):
   def __init__(self, e5_model):
       self.e5_model = e5_model
   def embed_documents(self, texts):
       return self.e5_model.embed_docs(texts).cpu().tolist()
   def embed_query(self, text):
       return self.e5_model.embed_queries([text]).cpu().tolist()[0]

if __name__ == "__main__":
    # Load E5 model
    model_name = os.getenv("E5_MODEL_NAME", "intfloat/e5-base-v2")
    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH", "model/checkpoints/model_epoch_1.pt")
    e5 = E5Model(model_name)
    e5.load_checkpoint(checkpoint_path)
    e5_embeddings = E5EmbeddingsWrapper(e5)

    # Load dataset
    data_path = os.getenv("DATASET_PATH", "data/val.json")
    with open(data_path, "r") as f:
        dataset = json.load(f)
    tools = [java_code_compiler]

    # Use get_llm to get the Ollama Llama-3 model
    llm = get_llm("llama3.1:8b-instruct-fp16")
    # Create and run BasicToolRagAlgorithm
    algo = BasicToolRagAlgorithm(settings={"top_k": 3, "embedding_model_id": model_name}, embedding_function=e5_embeddings)
    print("setting up")
    algo.set_up(llm, tools)
    print("setup complete")
    import asyncio
    async def main():
        # Prepare a QuerySpecification for the first query in your dataset
        query_spec = QuerySpecification(
            id=0,
            query=dataset[0]["query"],
            golden_tools={"java_code_compiler": {"tool_name": "java_code_compiler"}},
            additional_tools=None
        )
        response, relevant_tool_names = await algo.process_query(query_spec)
        print("process_query response:", response)
        print("relevant_tool_names:", relevant_tool_names)

    asyncio.run(main())
    algo.tear_down()



