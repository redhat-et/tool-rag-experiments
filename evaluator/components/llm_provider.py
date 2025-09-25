from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from evaluator.eval_spec import MODEL_ID_TO_URL, MODEL_ID_TO_PROVIDER_TYPE
from evaluator.utils.utils import print_verbose


def get_llm(model_id: str, **kwargs) -> BaseChatModel:
    """
    Initializes and returns an LLM wrapper object.
    The following LLM providers are currently supported: Ollama, vLLM, OpenAI.
    """
    if model_id not in MODEL_ID_TO_URL or model_id not in MODEL_ID_TO_PROVIDER_TYPE:
        raise ValueError(f"Unsupported model ID: {model_id}\n"
                         "Please make sure to add your model along with its URL "
                         "to MODEL_ID_TO_URL and MODEL_ID_TO_PROVIDER_TYPE in eval_spec.py")
    model_url = MODEL_ID_TO_URL[model_id]
    provider_id = MODEL_ID_TO_PROVIDER_TYPE[model_id]
    print_verbose(f"Connecting to {provider_id} server on {model_url} serving {model_id}...")
    if provider_id.lower() == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_id, base_url=model_url, **kwargs)
    if provider_id.lower() == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            openai_api_base=f"{model_url}/v1",
            openai_api_key="EMPTY",
            model=model_id,
            timeout=120,
            **kwargs
        )
    if provider_id.lower() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, **kwargs)

    raise ValueError(f"Unsupported provider: {provider_id}")


def query_llm(model: BaseChatModel, system_prompt: str, user_prompt: str) -> str:
    """
    Queries a given model with a given system and user prompts.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human", "{user}"),
    ])

    chain = prompt | model | StrOutputParser()
    return chain.invoke({"system": system_prompt, "user": user_prompt})
