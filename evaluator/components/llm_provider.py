from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_llm(provider_id: str, model_id: str, base_url: str, **kwargs) -> BaseChatModel:
    """
    Initializes and returns an LLM wrapper object.
    The following LLM providers are currently supported: Ollama, vLLM, OpenAI.
    """
    if provider_id.lower() == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_id, base_url=base_url, **kwargs)
    if provider_id.lower() == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(openai_api_base=f"{base_url}/v1", openai_api_key="EMPTY", model=model_id, **kwargs)
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
