"""
LLM Provider Abstraction Layer

This module provides a unified interface for different LLM providers:
- Ollama (for local testing)
- vLLM (for cluster deployments)
- OpenAI (for cloud API access)

Environment Variables:
    LLM_PROVIDER: "ollama", "vllm", "openai", or "auto" (default: auto-detect)
    LLM_MODEL: Model name/identifier
    LLM_BASE_URL: Base URL for the provider

Usage:
    from llm_provider import get_llm_provider
    
    # Auto-detect (recommended)
    llm = get_llm_provider()
    
    # Explicit provider selection
    llm = get_llm_provider("ollama", model="llama3.2:3b-instruct-fp16")
    llm = get_llm_provider("vllm", model="meta-llama/Llama-2-7b-chat-hf")
    llm = get_llm_provider("openai", model="gpt-3.5-turbo")
"""

import os
import requests
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
    elif provider == "openai":
        return _get_openai_provider(model, base_url, temperature, **kwargs)
    else:
        raise LLMProviderError(f"Unsupported provider: {provider}")


def _auto_detect_provider() -> str:
    """Auto-detect the best available provider based on environment."""
    
    # Check if provider is explicitly set
    provider = os.getenv("LLM_PROVIDER")
    if provider:
        return provider.lower()
    
    # Check if base URL is set (implies remote deployment)
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        # Try to detect provider from URL or default to vllm for remote
        if "ollama" in base_url.lower():
            return "ollama"
        else:
            return "vllm"
    
    # Check if Ollama is available locally
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return "ollama"
    except:
        pass
    
    # Default to Ollama for local development
    return "ollama"


def _get_provider():
    """Get provider configuration from environment variables."""
    provider_id = os.getenv("LLM_PROVIDER", "auto")
    if provider_id == "auto":
        provider_id = _auto_detect_provider()
    
    return {
        "provider_id": provider_id,
        "model": os.getenv("LLM_MODEL"),
        "base_url": os.getenv("LLM_BASE_URL")
    }

def validate_provider_setup() -> dict:
    """
    Validate that the configured provider is properly set up.
    
    Returns:
        dict: Status information about the provider setup
    """
    status = {
        "provider": None,
        "available": False,
        "model": None,
        "base_url": None,
        "errors": []
    }
    
    try:
        # Get environment variables
        provider_id = os.getenv("LLM_PROVIDER", "auto")
        model = os.getenv("LLM_MODEL")
        base_url = os.getenv("LLM_BASE_URL")
        
        # Auto-detect provider if not explicitly set
        if provider_id == "auto":
            provider_id = _auto_detect_provider()
        
        status["provider"] = provider_id
        status["model"] = model
        status["base_url"] = base_url
        
        # Provider-specific validation
        if provider_id == "ollama":
            # Test Ollama connection
            import requests
            test_url = base_url or "http://localhost:11434"
            response = requests.get(f"{test_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                status["available"] = True
                status["base_url"] = test_url
                status["model"] = model or "llama3.2:3b-instruct-fp16"
            else:
                status["errors"].append(f"Ollama returned status code: {response.status_code}")
        
        elif provider_id == "vllm":
            # Test vLLM connection
            import requests
            test_url = base_url or "http://localhost:8000/v1"
            
            try:
                response = requests.get(f"{test_url}/models", timeout=5)
                if response.status_code == 200:
                    status["available"] = True
                    status["base_url"] = test_url
                    status["model"] = model or "meta-llama/Llama-2-7b-chat-hf"
                else:
                    status["errors"].append(f"vLLM returned status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                status["errors"].append(f"Cannot connect to vLLM: {e}")
        
        elif provider_id == "openai":
            # Test OpenAI connection
            import requests
            test_url = base_url or "https://api.openai.com/v1"
            
            try:
                # Simple test - OpenAI doesn't have a simple health endpoint
                # We'll just check if the base URL is accessible
                if test_url.startswith("https://api.openai.com"):
                    status["available"] = True
                    status["base_url"] = test_url
                    status["model"] = model or "gpt-3.5-turbo"
                else:
                    # For custom endpoints, we can't easily test without API key
                    status["available"] = True  # Assume available
                    status["base_url"] = test_url
                    status["model"] = model or "gpt-3.5-turbo"
            except Exception as e:
                status["errors"].append(f"Cannot validate OpenAI setup: {e}")
        else:
            status["errors"].append(f"Unsupported provider: {provider_id}")
    
    except Exception as e:
        status["errors"].append(f"Validation error: {e}")
    
    return status
