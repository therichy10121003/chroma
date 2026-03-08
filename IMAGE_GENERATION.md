# Chroma Image Generation Features

## Overview

Chroma now includes comprehensive support for **AI-powered image generation and transformation**, combining the power of vector search with state-of-the-art image generation models like DALL-E and Stable Diffusion.

## Features

### 🎨 **Image Generative Functions**

Support for multiple image generation providers:
- **OpenAI DALL-E** (DALL-E 2, DALL-E 3)
- **Stable Diffusion** (Stability AI API, Replicate, Local)
- **Extensible protocol** for custom providers

### 🖌️ **Style Control**

20+ predefined visual styles:
- **Artistic**: Impressionist, Cubist, Surrealist, Art Nouveau
- **Photographic**: Cinematic, Portrait, Street Photography
- **Illustration**: Anime, Comic Book, Watercolor
- **Period**: Victorian, Art Deco
- **Genre**: Cyberpunk, Steampunk, Gothic, Sci-Fi

### 🎭 **Mood Control**

12+ emotional atmospheres:
- **Positive**: Cheerful, Romantic, Whimsical
- **Calm**: Serene, Meditative
- **Mysterious**: Ominous, Enigmatic
- **Energetic**: Dynamic, Epic

### ⚙️ **Advanced Features**

- Text-to-image generation
- Image-to-image transformation
- RAG-enhanced image prompts
- Multiple variation generation
- Quality and size control
- Style/mood compatibility scoring
- Comprehensive style/mood libraries

---

## Quick Start

### Installation

```bash
pip install chromadb openai stability-sdk
```

### Basic Usage

```python
import chromadb
from chromadb.utils.generative_functions import OpenAIImageGenerativeFunction

# Create image generator
img_gen = OpenAIImageGenerativeFunction(
    model="dall-e-3",
    size="1024x1024"
)

# Generate an image
image_url = img_gen(
    prompt="A serene mountain landscape",
    style="photorealistic",
    mood="serene"
)

print(f"Generated: {image_url}")
```

---

## API Reference

### ImageGenerativeFunction Protocol

All image generative functions implement this protocol:

```python
class ImageGenerativeFunction(Protocol):
    def __call__(
        self,
        prompt: str,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        **kwargs: Any
    ) -> ImageGenerativeOutput:
        """Generate an image from a text prompt."""
        ...

    def generate_variations(
        self,
        prompt: str,
        n: int = 3,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        **kwargs: Any
    ) -> List[ImageGenerativeOutput]:
        """Generate multiple variations."""
        ...

    def transform_image(
        self,
        image_input: Union[str, bytes],
        transformation_prompt: str,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        strength: float = 0.5,
        **kwargs: Any
    ) -> ImageGenerativeOutput:
        """Transform an existing image."""
        ...

    def apply_style(
        self,
        image_input: Union[str, bytes],
        style: str,
        intensity: float = 1.0,
        **kwargs: Any
    ) -> ImageGenerativeOutput:
        """Apply a specific style to an image."""
        ...

    def set_mood(
        self,
        image_input: Union[str, bytes],
        mood: str,
        intensity: float = 1.0,
        **kwargs: Any
    ) -> ImageGenerativeOutput:
        """Adjust the mood/atmosphere of an image."""
        ...
```

### OpenAIImageGenerativeFunction

```python
from chromadb.utils.generative_functions import OpenAIImageGenerativeFunction

img_gen = OpenAIImageGenerativeFunction(
    api_key=None,                    # Optional, uses OPENAI_API_KEY env var
    model="dall-e-3",                # "dall-e-2" or "dall-e-3"
    size="1024x1024",                # Image dimensions
    quality="standard",              # "standard" or "hd" (DALL-E 3 only)
    response_format="url",           # "url" or "b64_json"
)
```

**Supported Models:**
- `dall-e-3` - Latest, highest quality (1 image at a time)
- `dall-e-2` - Faster, can generate multiple images

**Supported Sizes:**
- DALL-E 2: `256x256`, `512x512`, `1024x1024`
- DALL-E 3: `1024x1024`, `1792x1024`, `1024x1792`

### StableDiffusionGenerativeFunction

```python
from chromadb.utils.generative_functions import StableDiffusionGenerativeFunction

sd_gen = StableDiffusionGenerativeFunction(
    api_key=None,                        # Optional, uses STABILITY_API_KEY
    api_provider="stability",            # "stability", "replicate", or "local"
    model="stable-diffusion-xl-1024-v1-0",  # Model ID
    width=1024,                          # Image width
    height=1024,                         # Image height
    steps=30,                            # Diffusion steps (10-50)
    guidance_scale=7.5,                  # Prompt adherence (1-20)
)
```

**API Providers:**
- `stability` - Stability AI official API (requires API key)
- `replicate` - Replicate platform (requires API key)
- `local` - Local Stable Diffusion (requires diffusers & torch)

### Collection.generate_image()

```python
result = collection.generate_image(
    prompt: str,                         # Required: Image description
    image_generative_function: ImageGenerativeFunction,  # Required
    style: Optional[str] = None,         # Visual style
    mood: Optional[str] = None,          # Emotional mood
    n_results: int = 3,                  # Documents to retrieve for context
    where: Optional[Where] = None,       # Metadata filter
    where_document: Optional[WhereDocument] = None,  # Document filter
    use_context: bool = True,            # Use RAG to enhance prompt
    **generation_kwargs                   # Additional parameters
)
```

**Returns:**
```python
{
    'image': str,                        # Generated image URL or base64
    'enhanced_prompt': str,              # Full prompt used for generation
    'source_documents': List[str],       # Documents used for context
    'style': str,                        # Applied style
    'mood': str,                         # Applied mood
}
```

### Collection.transform_image()

```python
result = collection.transform_image(
    image_input: Union[str, bytes],      # Source image
    transformation_prompt: str,          # How to transform
    image_generative_function: ImageGenerativeFunction,  # Required
    style: Optional[str] = None,         # Style to apply
    mood: Optional[str] = None,          # Mood to apply
    strength: float = 0.5,               # Transformation strength (0-1)
    **generation_kwargs
)
```

**Returns:**
```python
{
    'image': str,                        # Transformed image
    'transformation_prompt': str,        # Prompt used
    'style': str,                        # Applied style
    'mood': str,                         # Applied mood
    'strength': float,                   # Transformation strength
}
```

---

## Examples

### Example 1: Basic Text-to-Image

```python
from chromadb.utils.generative_functions import OpenAIImageGenerativeFunction

img_gen = OpenAIImageGenerativeFunction(model="dall-e-3")

image_url = img_gen(
    prompt="A magical forest with glowing mushrooms",
    style="fantasy",
    mood="mysterious"
)

print(f"Image URL: {image_url}")
```

### Example 2: RAG-Enhanced Image Generation

```python
import chromadb
from chromadb.utils.generative_functions import OpenAIImageGenerativeFunction

# Setup collection with art descriptions
client = chromadb.Client()
collection = client.create_collection(name="art_styles")

collection.add(
    documents=[
        "Impressionist art uses visible brush strokes and emphasis on light",
        "Cubism features geometric shapes and multiple perspectives",
        "Surrealism combines dreamlike and realistic elements"
    ],
    ids=["1", "2", "3"]
)

# Generate image using document context
img_gen = OpenAIImageGenerativeFunction(model="dall-e-3")

result = collection.generate_image(
    prompt="A beautiful garden",
    image_generative_function=img_gen,
    style="impressionist",
    mood="cheerful",
    n_results=2,  # Use top 2 documents for context
    use_context=True
)

print(f"Original prompt: A beautiful garden")
print(f"Enhanced prompt: {result['enhanced_prompt']}")
print(f"Image: {result['image']}")
```

### Example 3: Generate Multiple Variations

```python
img_gen = OpenAIImageGenerativeFunction(model="dall-e-2")

variations = img_gen.generate_variations(
    prompt="A cozy coffee shop",
    n=3,
    style="watercolor",
    mood="serene"
)

for i, url in enumerate(variations, 1):
    print(f"Variation {i}: {url}")
```

### Example 4: Image-to-Image Transformation

```python
result = collection.transform_image(
    image_input="my_photo.jpg",
    transformation_prompt="Make it look like a Van Gogh painting",
    image_generative_function=img_gen,
    style="impressionist",
    mood="artistic",
    strength=0.8  # High strength = more transformation
)

print(f"Transformed: {result['image']}")
```

### Example 5: Style Discovery

```python
from chromadb.utils.generative_functions import get_available_styles, get_available_moods

# Get all available styles
styles = get_available_styles()
print(f"Available styles: {styles}")

# Get all available moods
moods = get_available_moods()
print(f"Available moods: {moods}")

# Get style details
from chromadb.utils.generative_functions import StyleLibrary

style_profile = StyleLibrary.get_style("cyberpunk")
print(f"Cyberpunk style prompt: {style_profile.to_prompt()}")
```

### Example 6: Advanced Style/Mood Combinations

```python
from chromadb.utils.generative_functions import StyleMoodCombiner

# Get mood suggestions for a style
suggestions = StyleMoodCombiner.suggest_mood_for_style("impressionist", top_n=3)
print(f"Best moods for impressionist: {suggestions}")

# Create enhanced prompt
enhanced = StyleMoodCombiner.create_enhanced_prompt(
    base_prompt="A sunset over the ocean",
    style="cinematic",
    mood="romantic",
    enhance_quality=True
)
print(f"Enhanced prompt: {enhanced}")
```

### Example 7: Stable Diffusion with Fine Control

```python
from chromadb.utils.generative_functions import StableDiffusionGenerativeFunction

sd_gen = StableDiffusionGenerativeFunction(
    api_provider="stability",
    model="stable-diffusion-xl-1024-v1-0",
    width=1024,
    height=1024,
    steps=50,  # More steps = higher quality
    guidance_scale=8.5  # Higher = stricter prompt adherence
)

image_data = sd_gen(
    prompt="A majestic dragon flying over mountains",
    style="fantasy",
    mood="epic",
    steps=60,  # Override default
    guidance_scale=9.0
)

# image_data is base64 encoded
import base64
from PIL import Image
from io import BytesIO

image = Image.open(BytesIO(base64.b64decode(image_data)))
image.save("dragon.png")
```

### Example 8: Different Quality Levels

```python
# Standard quality (faster, cheaper)
img_gen_std = OpenAIImageGenerativeFunction(
    model="dall-e-3",
    quality="standard"
)

std_url = img_gen_std(prompt="A detailed cityscape", style="photorealistic")

# HD quality (slower, more expensive, higher quality)
img_gen_hd = OpenAIImageGenerativeFunction(
    model="dall-e-3",
    quality="hd"
)

hd_url = img_gen_hd(prompt="A detailed cityscape", style="photorealistic")

print(f"Standard: {std_url}")
print(f"HD: {hd_url}")
```

---

## Available Styles

### Artistic Styles
- `impressionist` - Visible brush strokes, emphasis on light and color
- `cubist` - Geometric shapes, multiple perspectives
- `surrealist` - Dreamlike, impossible scenes
- `art_nouveau` - Organic forms, decorative, flowing lines

### Photographic Styles
- `photorealistic` - Ultra detailed, 8k resolution
- `cinematic` - Film still, dramatic lighting
- `portrait_photography` - Professional portrait, studio lighting
- `street_photography` - Candid, documentary style

### Illustration Styles
- `anime` - Japanese animation style, cel shaded
- `comic_book` - Bold lines, halftone dots, dynamic
- `watercolor_illustration` - Soft edges, translucent colors
- `digital_art` - Modern digital illustration

### 3D and Technical
- `3d_render` - Blender, octane render, realistic lighting
- `pixel_art` - 8-bit, retro gaming style

### Genre Styles
- `cyberpunk` - Neon lights, futuristic, dystopian
- `steampunk` - Victorian mechanical, brass and copper
- `fantasy` - Magical, ethereal, concept art
- `gothic` - Dark, ornate, architectural
- `sci_fi` - Futuristic, advanced technology

---

## Available Moods

### Positive
- `cheerful` - Bright, happy, warm lighting
- `romantic` - Dreamy, intimate, soft focus
- `whimsical` - Playful, fantastical, lighthearted

### Calm
- `serene` - Peaceful, tranquil, soft lighting
- `meditative` - Zen, contemplative, still

### Mysterious
- `mysterious` - Enigmatic, foggy, secretive
- `ominous` - Foreboding, dark clouds, threatening

### Negative
- `melancholic` - Somber, muted colors, wistful

### Energetic
- `dynamic` - Powerful, intense, action-packed
- `epic` - Grand, majestic, awe-inspiring

---

## Best Practices

### 1. Choose the Right Model

- **DALL-E 3**: Best quality, more coherent text rendering
- **DALL-E 2**: Faster, cheaper, can generate multiple images
- **Stable Diffusion**: Most customizable, local option available

### 2. Optimize Prompts

```python
# Good: Descriptive with style/mood
img_gen(
    prompt="A Victorian mansion on a foggy hill",
    style="gothic",
    mood="ominous"
)

# Better: Add specific details
img_gen(
    prompt="A Victorian mansion with iron gates on a foggy hill, bare trees in foreground",
    style="gothic",
    mood="ominous"
)
```

### 3. Use RAG for Context-Aware Images

```python
# Add reference images or descriptions to your collection
collection.add(
    documents=["Art style references..."],
    ids=["ref1"]
)

# Generate with context
result = collection.generate_image(
    prompt="A portrait",
    image_generative_function=img_gen,
    use_context=True  # Enhances prompt with relevant docs
)
```

### 4. Experiment with Strength for Transformations

```python
# Subtle transformation (strength=0.3)
result = collection.transform_image(
    image_input="photo.jpg",
    transformation_prompt="Add autumn colors",
    style="photorealistic",
    strength=0.3  # Keep most of original
)

# Dramatic transformation (strength=0.9)
result = collection.transform_image(
    image_input="photo.jpg",
    transformation_prompt="Turn into abstract art",
    style="surrealist",
    strength=0.9  # Heavy transformation
)
```

### 5. Handle API Keys Securely

```bash
# Use environment variables
export OPENAI_API_KEY="sk-..."
export STABILITY_API_KEY="sk-..."
```

```python
# Don't hardcode keys
img_gen = OpenAIImageGenerativeFunction()  # Uses env var
```

---

## Style and Mood Library

### StyleLibrary Class

```python
from chromadb.utils.generative_functions import StyleLibrary

# Get a specific style
style = StyleLibrary.get_style("impressionist")
print(style.to_prompt())  # Get full prompt string

# List all styles
all_styles = StyleLibrary.list_styles()

# List by category
from chromadb.utils.generative_functions.image_style_algorithms import StyleCategory

artistic_styles = StyleLibrary.list_styles(category=StyleCategory.ARTISTIC)
```

### MoodLibrary Class

```python
from chromadb.utils.generative_functions import MoodLibrary

# Get a specific mood
mood = MoodLibrary.get_mood("serene")
print(mood.to_prompt())  # Get full prompt string

# List all moods
all_moods = MoodLibrary.list_moods()

# List by category
from chromadb.utils.generative_functions.image_style_algorithms import MoodCategory

positive_moods = MoodLibrary.list_moods(category=MoodCategory.POSITIVE)
```

### Style/Mood Compatibility

```python
from chromadb.utils.generative_functions import StyleMoodCombiner

# Check compatibility
score = StyleMoodCombiner.get_compatibility("cyberpunk", "ominous")
print(f"Compatibility: {score}")  # 0-1, higher = better match

# Get mood suggestions for a style
suggestions = StyleMoodCombiner.suggest_mood_for_style("watercolor_illustration")
# Returns: [("serene", 1.0), ("whimsical", 0.95), ...]

# Get style suggestions for a mood
suggestions = StyleMoodCombiner.suggest_style_for_mood("mysterious")
# Returns: [("surrealist", 0.9), ("gothic", 0.85), ...]
```

---

## Troubleshooting

### Issue: API Key Errors

```
ValueError: The OPENAI_API_KEY environment variable is not set.
```

**Solution:**
```bash
export OPENAI_API_KEY="your-key-here"
export STABILITY_API_KEY="your-key-here"
```

### Issue: Size Not Supported

```
ValueError: Invalid size 512x512 for dall-e-3
```

**Solution:**
Use correct sizes for each model:
- DALL-E 2: 256x256, 512x512, 1024x1024
- DALL-E 3: 1024x1024, 1792x1024, 1024x1792

### Issue: Image Transformation Not Supported

```
NotImplementedError: This image generative function does not support image transformation
```

**Solution:**
Use DALL-E 2 for transformations (DALL-E 3 doesn't support img2img):
```python
img_gen = OpenAIImageGenerativeFunction(model="dall-e-2")
```

Or use Stable Diffusion with img2img support.

---

## Performance Tips

### 1. Batch Generation

```python
# Generate multiple images efficiently
prompts = ["A cat", "A dog", "A bird"]
styles = ["anime", "photorealistic", "watercolor"]

images = []
for prompt, style in zip(prompts, styles):
    url = img_gen(prompt=prompt, style=style)
    images.append(url)
```

### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate(prompt: str, style: str) -> str:
    return img_gen(prompt=prompt, style=style)
```

### 3. Cost Optimization

```python
# Use DALL-E 2 for iteration (cheaper)
img_gen_draft = OpenAIImageGenerativeFunction(model="dall-e-2")
draft = img_gen_draft(prompt="A castle", style="fantasy")

# Use DALL-E 3 for final (higher quality)
img_gen_final = OpenAIImageGenerativeFunction(model="dall-e-3", quality="hd")
final = img_gen_final(prompt="A castle", style="fantasy")
```

---

## License

This feature is part of Chroma and follows the same Apache 2.0 license.

## Contributing

Contributions welcome! To add new image generation providers:

1. Implement the `ImageGenerativeFunction` protocol
2. Add to `chromadb/utils/generative_functions/`
3. Register in `__init__.py`
4. Add tests and examples

## Support

- **Documentation**: https://docs.trychroma.com
- **GitHub Issues**: https://github.com/chroma-core/chroma/issues
- **Discord**: https://discord.gg/chroma

---

**Happy creating! 🎨**
