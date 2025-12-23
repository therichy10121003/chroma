"""OpenAI DALL-E image generative function for text-to-image generation."""

from chromadb.api.types import ImageGenerativeFunction, ImageGenerativeOutput
from typing import Dict, Any, Optional, Union, List
import os
import warnings
import base64
from io import BytesIO


class OpenAIImageGenerativeFunction(ImageGenerativeFunction):
    """
    OpenAI DALL-E image generative function.

    Supports DALL-E 2 and DALL-E 3 for high-quality image generation with
    advanced style and mood control.
    """

    # Predefined style mappings for better results
    STYLE_MODIFIERS = {
        "photorealistic": "photorealistic, highly detailed, 8k resolution",
        "anime": "anime style, vibrant colors, Japanese animation",
        "oil painting": "oil painting, classical art style, brush strokes",
        "watercolor": "watercolor painting, soft colors, artistic",
        "digital art": "digital art, modern, detailed illustration",
        "3d render": "3D render, Blender, octane render, realistic lighting",
        "pixel art": "pixel art, 8-bit, retro gaming style",
        "sketch": "pencil sketch, hand-drawn, artistic sketch",
        "cyberpunk": "cyberpunk style, neon lights, futuristic, dark",
        "fantasy": "fantasy art, magical, ethereal, concept art",
        "minimalist": "minimalist design, clean, simple, modern",
        "impressionist": "impressionist painting, Monet style, artistic",
        "art nouveau": "art nouveau, decorative, ornate, vintage",
        "pop art": "pop art style, bold colors, Andy Warhol inspired",
    }

    # Mood modifiers for atmospheric control
    MOOD_MODIFIERS = {
        "cheerful": "bright, cheerful, happy atmosphere, warm lighting",
        "mysterious": "mysterious, enigmatic, shadowy, atmospheric",
        "dramatic": "dramatic lighting, intense, powerful composition",
        "serene": "serene, peaceful, calm, soft lighting",
        "melancholic": "melancholic, somber, muted colors, emotional",
        "energetic": "energetic, dynamic, vibrant, lively",
        "romantic": "romantic, dreamy, soft focus, warm tones",
        "ominous": "ominous, foreboding, dark atmosphere, moody",
        "whimsical": "whimsical, playful, fantastical, light-hearted",
        "epic": "epic, grand, cinematic, awe-inspiring",
        "nostalgic": "nostalgic, vintage feel, warm memories",
        "futuristic": "futuristic, sci-fi, advanced technology",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        api_key_env_var: str = "OPENAI_API_KEY",
        response_format: str = "url",
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI DALL-E image generative function.

        Args:
            api_key: OpenAI API key (not recommended for production)
            model: Model to use ("dall-e-2" or "dall-e-3")
            size: Image size ("1024x1024", "1792x1024", "1024x1792" for DALL-E 3)
            quality: Image quality ("standard" or "hd" for DALL-E 3)
            api_key_env_var: Environment variable for API key
            response_format: "url" or "b64_json"
            **kwargs: Additional parameters
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
                "Please use environment variables via api_key_env_var.",
                DeprecationWarning,
            )

        self.api_key_env_var = api_key_env_var
        self.api_key = api_key or os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"The {api_key_env_var} environment variable is not set. "
                "Please set it or provide api_key parameter."
            )

        self.model = model
        self.size = size
        self.quality = quality
        self.response_format = response_format
        self.kwargs = kwargs

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)

    def _enhance_prompt(
        self,
        prompt: str,
        style: Optional[str] = None,
        mood: Optional[str] = None,
    ) -> str:
        """
        Enhance the prompt with style and mood modifiers.

        Args:
            prompt: Base prompt
            style: Visual style
            mood: Emotional mood

        Returns:
            Enhanced prompt
        """
        enhanced = prompt

        # Add style modifiers
        if style:
            style_lower = style.lower()
            if style_lower in self.STYLE_MODIFIERS:
                enhanced += f", {self.STYLE_MODIFIERS[style_lower]}"
            else:
                enhanced += f", {style} style"

        # Add mood modifiers
        if mood:
            mood_lower = mood.lower()
            if mood_lower in self.MOOD_MODIFIERS:
                enhanced += f", {self.MOOD_MODIFIERS[mood_lower]}"
            else:
                enhanced += f", {mood} mood"

        return enhanced

    def __call__(
        self,
        prompt: str,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        **kwargs: Any,
    ) -> ImageGenerativeOutput:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image
            style: Visual style
            mood: Emotional tone
            **kwargs: Additional parameters (size, quality, n, etc.)

        Returns:
            Generated image URL or base64 data
        """
        # Enhance prompt with style and mood
        enhanced_prompt = self._enhance_prompt(prompt, style, mood)

        # Prepare parameters
        params: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "prompt": enhanced_prompt,
            "size": kwargs.get("size", self.size),
            "n": kwargs.get("n", 1),
        }

        # Add DALL-E 3 specific parameters
        if params["model"] == "dall-e-3":
            params["quality"] = kwargs.get("quality", self.quality)
            params["response_format"] = kwargs.get("response_format", self.response_format)

        # Generate image
        response = self.client.images.generate(**params)

        # Return first image
        if self.response_format == "url":
            return response.data[0].url or ""
        else:
            return response.data[0].b64_json or ""

    def generate_variations(
        self,
        prompt: str,
        n: int = 3,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        **kwargs: Any,
    ) -> List[ImageGenerativeOutput]:
        """
        Generate multiple variations of an image.

        Args:
            prompt: Text description
            n: Number of variations (max 10 for DALL-E 2, 1 for DALL-E 3)
            style: Visual style
            mood: Emotional tone
            **kwargs: Additional parameters

        Returns:
            List of generated images
        """
        enhanced_prompt = self._enhance_prompt(prompt, style, mood)

        params: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "prompt": enhanced_prompt,
            "size": kwargs.get("size", self.size),
            "n": n,
        }

        if params["model"] == "dall-e-3":
            # DALL-E 3 only supports n=1, so we generate multiple times
            if n > 1:
                warnings.warn(
                    "DALL-E 3 only supports n=1. Generating images sequentially."
                )
                return [self.__call__(prompt, style, mood, **kwargs) for _ in range(n)]
            params["quality"] = kwargs.get("quality", self.quality)

        response = self.client.images.generate(**params)

        if self.response_format == "url":
            return [img.url or "" for img in response.data]
        else:
            return [img.b64_json or "" for img in response.data]

    def transform_image(
        self,
        image_input: Union[str, bytes],
        transformation_prompt: str,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        strength: float = 0.5,
        **kwargs: Any,
    ) -> ImageGenerativeOutput:
        """
        Transform an existing image (DALL-E 2 edit feature).

        Args:
            image_input: Source image path or bytes
            transformation_prompt: How to transform the image
            style: Visual style
            mood: Mood
            strength: Transformation strength (not directly supported, used in prompt)
            **kwargs: Additional parameters

        Returns:
            Transformed image
        """
        enhanced_prompt = self._enhance_prompt(transformation_prompt, style, mood)

        # Prepare image file
        if isinstance(image_input, str):
            with open(image_input, "rb") as f:
                image_file = f.read()
        else:
            image_file = image_input

        # Create edit
        response = self.client.images.edit(
            model="dall-e-2",  # Only DALL-E 2 supports edits
            image=image_file,
            prompt=enhanced_prompt,
            n=1,
            size=kwargs.get("size", self.size),
        )

        if self.response_format == "url":
            return response.data[0].url or ""
        else:
            return response.data[0].b64_json or ""

    @staticmethod
    def name() -> str:
        """Return the name of the image generative function."""
        return "openai-dalle"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OpenAIImageGenerativeFunction":
        """Build from configuration."""
        return OpenAIImageGenerativeFunction(
            api_key_env_var=config.get("api_key_env_var", "OPENAI_API_KEY"),
            model=config.get("model", "dall-e-3"),
            size=config.get("size", "1024x1024"),
            quality=config.get("quality", "standard"),
            response_format=config.get("response_format", "url"),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return {
            "api_key_env_var": self.api_key_env_var,
            "model": self.model,
            "size": self.size,
            "quality": self.quality,
            "response_format": self.response_format,
        }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration."""
        if "model" in config:
            valid_models = ["dall-e-2", "dall-e-3"]
            if config["model"] not in valid_models:
                raise ValueError(
                    f"Invalid model: {config['model']}. "
                    f"Valid models: {valid_models}"
                )

        if "size" in config:
            valid_sizes = {
                "dall-e-2": ["256x256", "512x512", "1024x1024"],
                "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
            }
            model = config.get("model", "dall-e-3")
            if config["size"] not in valid_sizes.get(model, []):
                raise ValueError(
                    f"Invalid size {config['size']} for {model}. "
                    f"Valid sizes: {valid_sizes.get(model, [])}"
                )

    def get_available_styles(self) -> List[str]:
        """Get list of available predefined styles."""
        return list(self.STYLE_MODIFIERS.keys())

    def get_available_moods(self) -> List[str]:
        """Get list of available predefined moods."""
        return list(self.MOOD_MODIFIERS.keys())
