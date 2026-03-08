"""Anthropic (Claude) generative function for text generation and RAG."""

from chromadb.api.types import GenerativeFunction
from typing import List, Dict, Any, Optional
import os
import warnings


class AnthropicGenerativeFunction(GenerativeFunction):
    """
    Anthropic (Claude) generative function for text generation.

    Supports Claude models for generating responses with optional context
    from retrieved documents (RAG).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        api_key_env_var: str = "ANTHROPIC_API_KEY",
        **kwargs: Any,
    ):
        """
        Initialize the Anthropic generative function.

        Args:
            api_key: Anthropic API key (not recommended for production)
            model_name: The Claude model to use (e.g., "claude-3-5-sonnet-20241022")
            temperature: Sampling temperature (0-1, higher = more creative)
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt for the model
            api_key_env_var: Environment variable for API key
            **kwargs: Additional parameters passed to Anthropic API
        """
        try:
            import anthropic
        except ImportError:
            raise ValueError(
                "The anthropic python package is not installed. "
                "Please install it with `pip install anthropic`"
            )

        if api_key is not None:
            warnings.warn(
                "Direct api_key configuration will not be persisted. "
                "Please use environment variables via api_key_env_var for persistent storage.",
                DeprecationWarning,
            )

        self.api_key_env_var = api_key_env_var
        self.api_key = api_key or os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"The {api_key_env_var} environment variable is not set. "
                "Please set it or provide api_key parameter."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. Answer questions based on the provided context. "
            "If the context doesn't contain relevant information, say so clearly."
        )
        self.kwargs = kwargs

        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def __call__(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response given a prompt and optional context.

        Args:
            prompt: The user's query/prompt
            context: Optional list of context documents for RAG
            **kwargs: Override default parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response
        """
        # Build the user message
        user_message = prompt

        # Add context if provided (RAG pattern)
        if context and len(context) > 0:
            context_text = "\n\n".join(
                [f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)]
            )
            user_message = (
                f"<context>\n{context_text}\n</context>\n\n"
                f"<question>\n{prompt}\n</question>\n\n"
                "Please answer the question based on the context provided above."
            )

        # Merge default params with overrides
        params = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        params.update({k: v for k, v in kwargs.items() if k not in params})

        # Generate response
        response = self.client.messages.create(**params)

        # Extract text from response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""

    def generate_batch(
        self,
        prompts: List[str],
        contexts: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batch.

        Note: Anthropic also supports batch API for cost savings.
        This implementation uses sequential calls for simplicity.

        Args:
            prompts: List of prompts
            contexts: Optional list of context document lists
            **kwargs: Generation parameters

        Returns:
            List of generated responses
        """
        if contexts is None:
            contexts = [None] * len(prompts)  # type: ignore

        return [
            self.__call__(prompt, context, **kwargs)
            for prompt, context in zip(prompts, contexts)
        ]

    @staticmethod
    def name() -> str:
        """Return the name of the generative function."""
        return "anthropic"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "AnthropicGenerativeFunction":
        """
        Build the generative function from a config.

        Args:
            config: Configuration dictionary

        Returns:
            Initialized AnthropicGenerativeFunction
        """
        return AnthropicGenerativeFunction(
            api_key_env_var=config.get("api_key_env_var", "ANTHROPIC_API_KEY"),
            model_name=config.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1024),
            system_prompt=config.get("system_prompt"),
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config for serialization.

        Returns:
            Configuration dictionary
        """
        return {
            "api_key_env_var": self.api_key_env_var,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
        }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate the configuration."""
        if "model_name" in config:
            valid_prefixes = ["claude-3", "claude-2"]
            if not any(config["model_name"].startswith(p) for p in valid_prefixes):
                warnings.warn(
                    f"Model {config['model_name']} may not be a valid Anthropic model. "
                    f"Expected models starting with: {valid_prefixes}"
                )
