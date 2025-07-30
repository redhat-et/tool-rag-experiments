"""
LLM Provider Abstraction Layer

This module provides a unified interface for different LLM providers:
- Ollama (for local testing)
- vLLM (for cluster deployments)

Usage:
    from llm_provider import get_llm_provider
    
    # For local testing with Ollama
    llm = get_llm_provider("ollama", model="llama3.2:3b-instruct-fp16")
    
    # For cluster deployment with vLLM
    llm = get_llm_provider("vllm", model="meta-llama/Llama-2-7b-chat-hf", base_url="http://cluster:8000/v1")
"""

import os
from typing import Optional, Union
from dotenv import load_dotenv

load_dotenv()

class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""
    pass

def get_llm_provider(
    provider: str = "auto",
    model: str = None,
    base_url: Optional[str] = None,
    temperature: float = 0,
    **kwargs
):
    """
    Get an LLM provider instance.
    
    Args:
        provider: "ollama", "vllm", or "auto" (auto-detect based on environment)
        model: Model name/identifier
        base_url: Base URL for the provider (optional, uses defaults)
        temperature: Sampling temperature
        **kwargs: Additional provider-specific arguments
    
    Returns:
        LLM instance compatible with LangChain
    
    Raises:
        LLMProviderError: If provider is not supported or configuration is invalid
    """
    
    # Auto-detect provider if not specified
    if provider == "auto":
        provider = _auto_detect_provider()
    
    provider = provider.lower()
    
    if provider == "ollama":
        return _get_ollama_provider(model, base_url, temperature, **kwargs)
    elif provider == "vllm":
        return _get_vllm_provider(model, base_url, temperature, **kwargs)
    else:
        raise LLMProviderError(f"Unsupported provider: {provider}")

def _auto_detect_provider() -> str:
    """Auto-detect the best available provider based on environment."""
    
    # Check for vLLM environment variables (cluster deployment)
    if os.getenv("VLLM_BASE_URL") or os.getenv("VLLM_MODEL"):
        return "vllm"
    
    # Check if Ollama is available locally
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return "ollama"
    except:
        pass
    
    # Default to Ollama for local development
    return "ollama"

def _get_ollama_provider(model: str, base_url: Optional[str], temperature: float, **kwargs):
    """Get Ollama LLM provider."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise LLMProviderError("langchain-ollama not installed. Run: pip install langchain-ollama")
    
    if not model:
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-fp16")
    
    if not base_url:
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs
    )

def _get_vllm_provider(model: str, base_url: Optional[str], temperature: float, **kwargs):
    """Get vLLM LLM provider."""
    try:
        from langchain_community.llms import VLLM
    except ImportError:
        raise LLMProviderError("langchain-community not installed. Run: pip install langchain-community")
    
    if not model:
        model = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    
    if not base_url:
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    
    # vLLM specific configuration
    vllm_kwargs = {
        "trust_remote_code": True,
        "max_new_tokens": 512,
        "top_p": 0.95,
        "temperature": temperature,
        **kwargs
    }
    
    return VLLM(
        model=model,
        endpoint=base_url,
        **vllm_kwargs
    )

def validate_provider_setup(provider: str = "auto") -> dict:
    """
    Validate that the specified provider is properly configured.
    
    Returns:
        dict: Status information about the provider setup
    """
    status = {
        "provider": provider,
        "available": False,
        "model": None,
        "base_url": None,
        "errors": []
    }
    
    try:
        if provider == "auto":
            provider = _auto_detect_provider()
            status["provider"] = provider
        
        if provider == "ollama":
            # Test Ollama connection
            import requests
            base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                status["available"] = True
                status["base_url"] = base_url
                status["model"] = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-fp16")
            else:
                status["errors"].append(f"Ollama returned status code: {response.status_code}")
        
        elif provider == "vllm":
            # Test vLLM connection
            import requests
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            
            try:
                response = requests.get(f"{base_url}/models", timeout=5)
                if response.status_code == 200:
                    status["available"] = True
                    status["base_url"] = base_url
                    status["model"] = os.getenv("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
                else:
                    status["errors"].append(f"vLLM returned status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                status["errors"].append(f"Cannot connect to vLLM: {e}")
    
    except Exception as e:
        status["errors"].append(f"Validation error: {e}")
    
    return status

# Convenience functions for common use cases
def get_local_llm(model: str = "llama3.2:3b-instruct-fp16", **kwargs):
    """Get LLM for local testing (Ollama)."""
    return get_llm_provider("ollama", model=model, **kwargs)

def get_cluster_llm(model: str = None, base_url: str = None, **kwargs):
    """Get LLM for cluster deployment (vLLM)."""
    return get_llm_provider("vllm", model=model, base_url=base_url, **kwargs) 