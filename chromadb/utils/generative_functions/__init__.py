"""Generative functions for text generation and RAG (Retrieval-Augmented Generation)."""

from typing import Dict, Any, Type, Set
from chromadb.api.types import GenerativeFunction

# Import all generative functions
from chromadb.utils.generative_functions.openai_generative import (
    OpenAIGenerativeFunction,
)
from chromadb.utils.generative_functions.anthropic_generative import (
    AnthropicGenerativeFunction,
)

# Get all the class names for backward compatibility
_all_classes: Set[str] = {
    "OpenAIGenerativeFunction",
    "AnthropicGenerativeFunction",
}

__all__ = list(_all_classes)


# Registry for generative function types
_GENERATIVE_FUNCTION_REGISTRY: Dict[str, Type[GenerativeFunction]] = {
    "openai": OpenAIGenerativeFunction,
    "anthropic": AnthropicGenerativeFunction,
}


def get_generative_function(name: str) -> Type[GenerativeFunction]:
    """
    Get a generative function class by name.

    Args:
        name: The name of the generative function (e.g., "openai", "anthropic")

    Returns:
        The generative function class

    Raises:
        ValueError: If the generative function name is not recognized
    """
    if name not in _GENERATIVE_FUNCTION_REGISTRY:
        raise ValueError(
            f"Unknown generative function: {name}. "
            f"Available functions: {list(_GENERATIVE_FUNCTION_REGISTRY.keys())}"
        )
    return _GENERATIVE_FUNCTION_REGISTRY[name]


def register_generative_function(name: str, func_class: Type[GenerativeFunction]) -> None:
    """
    Register a custom generative function.

    Args:
        name: The name to register the function under
        func_class: The generative function class to register
    """
    _GENERATIVE_FUNCTION_REGISTRY[name] = func_class
