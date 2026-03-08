"""Generative functions for text generation and RAG (Retrieval-Augmented Generation)."""

from typing import Dict, Any, Type, Set
from chromadb.api.types import GenerativeFunction, ImageGenerativeFunction

# Import all text generative functions
from chromadb.utils.generative_functions.openai_generative import (
    OpenAIGenerativeFunction,
)
from chromadb.utils.generative_functions.anthropic_generative import (
    AnthropicGenerativeFunction,
)

# Import all image generative functions
from chromadb.utils.generative_functions.openai_image_generative import (
    OpenAIImageGenerativeFunction,
)
from chromadb.utils.generative_functions.stable_diffusion_generative import (
    StableDiffusionGenerativeFunction,
)

# Import style and mood utilities
from chromadb.utils.generative_functions.image_style_algorithms import (
    StyleLibrary,
    MoodLibrary,
    StyleMoodCombiner,
    get_available_styles,
    get_available_moods,
    enhance_prompt_with_style_mood,
)

# Get all the class names for backward compatibility
_all_classes: Set[str] = {
    "OpenAIGenerativeFunction",
    "AnthropicGenerativeFunction",
    "OpenAIImageGenerativeFunction",
    "StableDiffusionGenerativeFunction",
    "StyleLibrary",
    "MoodLibrary",
    "StyleMoodCombiner",
}

__all__ = list(_all_classes) + [
    "get_available_styles",
    "get_available_moods",
    "enhance_prompt_with_style_mood",
]


# Registry for text generative function types
_GENERATIVE_FUNCTION_REGISTRY: Dict[str, Type[GenerativeFunction]] = {
    "openai": OpenAIGenerativeFunction,
    "anthropic": AnthropicGenerativeFunction,
}

# Registry for image generative function types
_IMAGE_GENERATIVE_FUNCTION_REGISTRY: Dict[str, Type[ImageGenerativeFunction]] = {
    "openai-dalle": OpenAIImageGenerativeFunction,
    "stable-diffusion": StableDiffusionGenerativeFunction,
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


def get_image_generative_function(name: str) -> Type[ImageGenerativeFunction]:
    """
    Get an image generative function class by name.

    Args:
        name: The name of the image generative function (e.g., "openai-dalle", "stable-diffusion")

    Returns:
        The image generative function class

    Raises:
        ValueError: If the image generative function name is not recognized
    """
    if name not in _IMAGE_GENERATIVE_FUNCTION_REGISTRY:
        raise ValueError(
            f"Unknown image generative function: {name}. "
            f"Available functions: {list(_IMAGE_GENERATIVE_FUNCTION_REGISTRY.keys())}"
        )
    return _IMAGE_GENERATIVE_FUNCTION_REGISTRY[name]


def register_image_generative_function(
    name: str, func_class: Type[ImageGenerativeFunction]
) -> None:
    """
    Register a custom image generative function.

    Args:
        name: The name to register the function under
        func_class: The image generative function class to register
    """
    _IMAGE_GENERATIVE_FUNCTION_REGISTRY[name] = func_class
