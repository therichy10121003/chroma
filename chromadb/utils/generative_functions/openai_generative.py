"""OpenAI generative function for text generation and RAG."""

from chromadb.api.types import GenerativeFunction
from typing import List, Dict, Any, Optional
import os
import warnings


class OpenAIGenerativeFunction(GenerativeFunction):
    """
    OpenAI generative function for text generation.

    Supports GPT-3.5, GPT-4, and other OpenAI chat models for generating
    responses with optional context from retrieved documents (RAG).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        api_key_env_var: str = "OPENAI_API_KEY",
        organization_id: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI generative function.

        Args:
            api_key: OpenAI API key (not recommended for production)
            model_name: The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Sampling temperature (0-2, higher = more creative)
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt for the model
            api_key_env_var: Environment variable for API key
            organization_id: Optional OpenAI organization ID
            api_base: Optional custom API base URL
            **kwargs: Additional parameters passed to OpenAI API
        """
        try:
            import openai
        except ImportError:
            raise ValueError(
                "The openai python package is not installed. "
                "Please install it with `pip install openai`"
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
            "If the context doesn't contain relevant information, say so."
        )
        self.organization_id = organization_id
        self.api_base = api_base
        self.kwargs = kwargs

        # Initialize the OpenAI client
        client_params: Dict[str, Any] = {"api_key": self.api_key}
        if self.organization_id is not None:
            client_params["organization"] = self.organization_id
        if self.api_base is not None:
            client_params["base_url"] = self.api_base

        self.client = openai.OpenAI(**client_params)

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
        # Build messages
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add context if provided (RAG pattern)
        if context and len(context) > 0:
            context_text = "\n\n".join(
                [f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)]
            )
            augmented_prompt = (
                f"Context:\n{context_text}\n\n"
                f"Question: {prompt}\n\n"
                "Answer based on the context provided above."
            )
            messages.append({"role": "user", "content": augmented_prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        # Merge default params with overrides
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        params.update({k: v for k, v in kwargs.items() if k not in params})

        # Generate response
        response = self.client.chat.completions.create(**params)

        return response.choices[0].message.content or ""

    def generate_batch(
        self,
        prompts: List[str],
        contexts: Optional[List[List[str]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batch.

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
        return "openai"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OpenAIGenerativeFunction":
        """
        Build the generative function from a config.

        Args:
            config: Configuration dictionary

        Returns:
            Initialized OpenAIGenerativeFunction
        """
        return OpenAIGenerativeFunction(
            api_key_env_var=config.get("api_key_env_var", "OPENAI_API_KEY"),
            model_name=config.get("model_name", "gpt-4"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 500),
            system_prompt=config.get("system_prompt"),
            organization_id=config.get("organization_id"),
            api_base=config.get("api_base"),
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
            "organization_id": self.organization_id,
            "api_base": self.api_base,
        }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate the configuration."""
        if "model_name" in config:
            valid_prefixes = ["gpt-3.5", "gpt-4", "gpt-4o"]
            if not any(config["model_name"].startswith(p) for p in valid_prefixes):
                warnings.warn(
                    f"Model {config['model_name']} may not be a valid OpenAI chat model. "
                    f"Expected models starting with: {valid_prefixes}"
                )
