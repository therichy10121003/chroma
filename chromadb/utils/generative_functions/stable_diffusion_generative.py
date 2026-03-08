"""Stable Diffusion image generative function via various APIs."""

from chromadb.api.types import ImageGenerativeFunction, ImageGenerativeOutput
from typing import Dict, Any, Optional, Union, List
import os
import warnings


class StableDiffusionGenerativeFunction(ImageGenerativeFunction):
    """
    Stable Diffusion image generative function.

    Supports various Stable Diffusion APIs including Stability AI, Replicate,
    and local installations. Provides advanced control over style, mood,
    and generation parameters.
    """

    # Style presets optimized for Stable Diffusion
    STYLE_PROMPTS = {
        "photorealistic": "photorealistic, ultra detailed, 8k uhd, high quality, sharp focus",
        "anime": "anime style, cel shaded, vibrant colors, studio ghibli inspired",
        "oil painting": "oil painting, classical art, masterpiece, detailed brush strokes",
        "watercolor": "watercolor painting, soft colors, artistic, flowing",
        "digital art": "digital art, detailed illustration, artstation trending",
        "3d render": "3d render, octane render, unreal engine, ray tracing",
        "pixel art": "pixel art, 16-bit, retro game style, crisp pixels",
        "sketch": "pencil sketch, hand drawn, artistic, detailed linework",
        "cyberpunk": "cyberpunk, neon lights, futuristic city, blade runner style",
        "fantasy": "fantasy art, magical, ethereal, d&d style, concept art",
        "steampunk": "steampunk, Victorian era, brass and copper, mechanical",
        "gothic": "gothic art, dark, ornate, architectural details",
        "sci-fi": "science fiction, futuristic, advanced technology, spaceship",
        "surreal": "surrealist art, dreamlike, Salvador Dali inspired, abstract",
        "comic book": "comic book style, bold lines, halftone dots, dynamic",
        "film noir": "film noir, black and white, high contrast, dramatic shadows",
    }

    # Mood/atmosphere presets
    MOOD_PROMPTS = {
        "cheerful": "cheerful, bright, happy, warm lighting, uplifting",
        "mysterious": "mysterious, enigmatic, foggy, shadowy atmosphere",
        "dramatic": "dramatic lighting, intense, powerful, cinematic",
        "serene": "serene, peaceful, calm, tranquil, soft light",
        "melancholic": "melancholic, somber, muted colors, emotional, rain",
        "energetic": "energetic, dynamic, vibrant, action-packed, motion",
        "romantic": "romantic, dreamy, soft focus, warm golden hour",
        "ominous": "ominous, foreboding, dark clouds, stormy, threatening",
        "whimsical": "whimsical, playful, fantastical, colorful, fun",
        "epic": "epic scale, grand, cinematic composition, awe-inspiring",
        "nostalgic": "nostalgic, vintage, faded colors, old photograph",
        "ethereal": "ethereal, otherworldly, glowing, magical atmosphere",
    }

    # Negative prompts to improve quality
    DEFAULT_NEGATIVE = (
        "ugly, tiling, poorly drawn, bad anatomy, bad proportions, "
        "blurry, duplicate, deformed, disfigured, low quality, jpeg artifacts"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_provider: str = "stability",
        model: str = "stable-diffusion-xl-1024-v1-0",
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance_scale: float = 7.5,
        api_key_env_var: str = "STABILITY_API_KEY",
        negative_prompt: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize Stable Diffusion generative function.

        Args:
            api_key: API key (not recommended for production)
            api_provider: API provider ("stability", "replicate", "local")
            model: Model ID to use
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of diffusion steps (10-50)
            guidance_scale: How closely to follow prompt (1-20, higher = stricter)
            api_key_env_var: Environment variable for API key
            negative_prompt: Things to avoid in generation
            **kwargs: Additional parameters
        """
        if api_key is not None:
            warnings.warn(
                "Direct api_key configuration will not be persisted.",
                DeprecationWarning,
            )

        self.api_key_env_var = api_key_env_var
        self.api_key = api_key or os.getenv(api_key_env_var)
        self.api_provider = api_provider
        self.model = model
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt or self.DEFAULT_NEGATIVE
        self.kwargs = kwargs

        # Initialize API client based on provider
        if api_provider == "stability":
            self._init_stability_client()
        elif api_provider == "replicate":
            self._init_replicate_client()
        elif api_provider == "local":
            self._init_local_client()
        else:
            raise ValueError(
                f"Unknown API provider: {api_provider}. "
                "Supported: stability, replicate, local"
            )

    def _init_stability_client(self) -> None:
        """Initialize Stability AI client."""
        try:
            import stability_sdk
        except ImportError:
            raise ValueError(
                "stability-sdk not installed. "
                "Install with: pip install stability-sdk"
            )

        if not self.api_key:
            raise ValueError(
                f"API key required. Set {self.api_key_env_var} environment variable."
            )

        # Client will be initialized on first use
        self.client = None

    def _init_replicate_client(self) -> None:
        """Initialize Replicate client."""
        try:
            import replicate
        except ImportError:
            raise ValueError(
                "replicate not installed. "
                "Install with: pip install replicate"
            )

        if not self.api_key:
            raise ValueError("REPLICATE_API_TOKEN environment variable required.")

        self.client = replicate

    def _init_local_client(self) -> None:
        """Initialize local Stable Diffusion client."""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
        except ImportError:
            raise ValueError(
                "diffusers and torch not installed. "
                "Install with: pip install diffusers torch"
            )

        # Load pipeline
        self.client = StableDiffusionPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16,
        )

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client = self.client.to(device)

    def _enhance_prompt(
        self,
        prompt: str,
        style: Optional[str] = None,
        mood: Optional[str] = None,
    ) -> str:
        """Enhance prompt with style and mood."""
        enhanced = prompt

        if style:
            style_lower = style.lower()
            if style_lower in self.STYLE_PROMPTS:
                enhanced += f", {self.STYLE_PROMPTS[style_lower]}"
            else:
                enhanced += f", {style} style"

        if mood:
            mood_lower = mood.lower()
            if mood_lower in self.MOOD_PROMPTS:
                enhanced += f", {self.MOOD_PROMPTS[mood_lower]}"
            else:
                enhanced += f", {mood} atmosphere"

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
            prompt: Text description
            style: Visual style
            mood: Emotional tone
            **kwargs: Additional parameters

        Returns:
            Generated image (URL or bytes)
        """
        enhanced_prompt = self._enhance_prompt(prompt, style, mood)

        if self.api_provider == "stability":
            return self._generate_stability(enhanced_prompt, **kwargs)
        elif self.api_provider == "replicate":
            return self._generate_replicate(enhanced_prompt, **kwargs)
        elif self.api_provider == "local":
            return self._generate_local(enhanced_prompt, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.api_provider}")

    def _generate_stability(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerativeOutput:
        """Generate using Stability AI API."""
        import requests

        url = "https://api.stability.ai/v1/generation/{}/text-to-image".format(
            kwargs.get("model", self.model)
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "text_prompts": [
                {"text": prompt, "weight": 1},
                {"text": kwargs.get("negative_prompt", self.negative_prompt), "weight": -1},
            ],
            "cfg_scale": kwargs.get("guidance_scale", self.guidance_scale),
            "height": kwargs.get("height", self.height),
            "width": kwargs.get("width", self.width),
            "samples": kwargs.get("samples", 1),
            "steps": kwargs.get("steps", self.steps),
        }

        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()

        data = response.json()
        # Return base64 image data
        return data["artifacts"][0]["base64"]

    def _generate_replicate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerativeOutput:
        """Generate using Replicate API."""
        output = self.client.run(
            kwargs.get("model", "stability-ai/sdxl:latest"),
            input={
                "prompt": prompt,
                "negative_prompt": kwargs.get("negative_prompt", self.negative_prompt),
                "width": kwargs.get("width", self.width),
                "height": kwargs.get("height", self.height),
                "num_inference_steps": kwargs.get("steps", self.steps),
                "guidance_scale": kwargs.get("guidance_scale", self.guidance_scale),
            },
        )

        # Return first output URL
        return output[0] if isinstance(output, list) else output

    def _generate_local(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerativeOutput:
        """Generate using local Stable Diffusion."""
        image = self.client(
            prompt,
            negative_prompt=kwargs.get("negative_prompt", self.negative_prompt),
            num_inference_steps=kwargs.get("steps", self.steps),
            guidance_scale=kwargs.get("guidance_scale", self.guidance_scale),
            height=kwargs.get("height", self.height),
            width=kwargs.get("width", self.width),
        ).images[0]

        # Convert to bytes
        from io import BytesIO
        import base64

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

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
        Transform an image using img2img.

        Args:
            image_input: Source image
            transformation_prompt: How to transform
            style: Visual style
            mood: Mood
            strength: How much to change (0-1)
            **kwargs: Additional parameters

        Returns:
            Transformed image
        """
        enhanced_prompt = self._enhance_prompt(transformation_prompt, style, mood)

        if self.api_provider == "local":
            from diffusers import StableDiffusionImg2ImgPipeline
            from PIL import Image
            import torch

            # Load img2img pipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)

            # Load input image
            if isinstance(image_input, str):
                init_image = Image.open(image_input).convert("RGB")
            else:
                from io import BytesIO
                init_image = Image.open(BytesIO(image_input)).convert("RGB")

            # Transform
            output = pipe(
                prompt=enhanced_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=kwargs.get("guidance_scale", self.guidance_scale),
                num_inference_steps=kwargs.get("steps", self.steps),
            ).images[0]

            # Convert to base64
            from io import BytesIO
            import base64

            buffered = BytesIO()
            output.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        raise NotImplementedError(
            f"Image transformation not yet implemented for {self.api_provider}"
        )

    @staticmethod
    def name() -> str:
        """Return function name."""
        return "stable-diffusion"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "StableDiffusionGenerativeFunction":
        """Build from configuration."""
        return StableDiffusionGenerativeFunction(
            api_key_env_var=config.get("api_key_env_var", "STABILITY_API_KEY"),
            api_provider=config.get("api_provider", "stability"),
            model=config.get("model", "stable-diffusion-xl-1024-v1-0"),
            width=config.get("width", 1024),
            height=config.get("height", 1024),
            steps=config.get("steps", 30),
            guidance_scale=config.get("guidance_scale", 7.5),
            negative_prompt=config.get("negative_prompt"),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            "api_key_env_var": self.api_key_env_var,
            "api_provider": self.api_provider,
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
        }

    def get_available_styles(self) -> List[str]:
        """Get available predefined styles."""
        return list(self.STYLE_PROMPTS.keys())

    def get_available_moods(self) -> List[str]:
        """Get available predefined moods."""
        return list(self.MOOD_PROMPTS.keys())
