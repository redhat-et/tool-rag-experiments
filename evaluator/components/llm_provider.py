from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from evaluator.config.schema import ModelConfig, ProviderId
from evaluator.utils.utils import log_verbose


def get_llm(model_id: str, model_config: List[ModelConfig], **kwargs) -> BaseChatModel:
    """
    Initializes and returns an LLM wrapper object.
    The following LLM providers are currently supported: Ollama, vLLM, OpenAI.
    """
    relevant_model_configs = [mc for mc in model_config if mc.id == model_id]
    if not relevant_model_configs:
        raise ValueError(f"Unsupported model ID: {model_id}\n"
                         "Please make sure to register your model along with its URL in the configuration file.")
    config = relevant_model_configs[0]

    log_verbose(f"Connecting to {config.provider_id} server on {config.url} serving {model_id}...")
    stripped_url = str(config.url).strip('/')
    if config.provider_id == ProviderId.OLLAMA:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_id, base_url=stripped_url, **kwargs)
    if config.provider_id == ProviderId.VLLM:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            openai_api_base=f"{stripped_url}/v1",
            openai_api_key="EMPTY",
            model=model_id,
            timeout=120,
            **kwargs
        )
    if config.provider_id == ProviderId.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, **kwargs)

    raise ValueError(f"Unsupported provider: {config.provider_id}")


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
