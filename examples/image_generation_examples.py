"""
Chroma Image Generation Features Examples
==========================================

This file demonstrates how to use the new image generation features in Chroma,
including text-to-image, image-to-image transformation, and advanced style/mood control.

Requirements:
    pip install chromadb openai stability-sdk diffusers torch

Environment variables:
    OPENAI_API_KEY - Your OpenAI API key (for DALL-E)
    STABILITY_API_KEY - Your Stability AI API key (for Stable Diffusion)
"""

import chromadb
from chromadb.utils.generative_functions import (
    OpenAIImageGenerativeFunction,
    StableDiffusionGenerativeFunction,
    get_available_styles,
    get_available_moods,
)
import os


def example_1_basic_image_generation():
    """Example 1: Basic image generation with DALL-E."""
    print("=" * 80)
    print("Example 1: Basic Image Generation with DALL-E")
    print("=" * 80)

    # Create image generator
    img_gen = OpenAIImageGenerativeFunction(
        model="dall-e-3",
        size="1024x1024",
        quality="standard"
    )

    # Generate an image
    image_url = img_gen(
        prompt="A serene mountain landscape at sunset",
        style="photorealistic",
        mood="serene"
    )

    print(f"\nPrompt: A serene mountain landscape at sunset")
    print(f"Style: photorealistic")
    print(f"Mood: serene")
    print(f"Generated Image URL: {image_url}")


def example_2_style_variations():
    """Example 2: Generate same scene with different styles."""
    print("\n" + "=" * 80)
    print("Example 2: Style Variations")
    print("=" * 80)

    img_gen = OpenAIImageGenerativeFunction(model="dall-e-3")

    prompt = "A futuristic cityscape with flying cars"
    styles = ["cyberpunk", "anime", "oil painting", "3d render"]

    print(f"\nGenerating '{prompt}' in different styles:\n")

    for style in styles:
        image_url = img_gen(
            prompt=prompt,
            style=style,
            mood="dynamic"
        )
        print(f"  [{style}] -> {image_url[:60]}...")


def example_3_mood_control():
    """Example 3: Same scene with different moods."""
    print("\n" + "=" * 80)
    print("Example 3: Mood Control")
    print("=" * 80)

    img_gen = OpenAIImageGenerativeFunction(model="dall-e-3")

    prompt = "A forest path"
    moods = ["cheerful", "mysterious", "ominous", "serene"]

    print(f"\nGenerating '{prompt}' with different moods:\n")

    for mood in moods:
        image_url = img_gen(
            prompt=prompt,
            style="photorealistic",
            mood=mood
        )
        print(f"  [{mood}] -> {image_url[:60]}...")


def example_4_rag_enhanced_image_generation():
    """Example 4: Use RAG to enhance image prompts from documents."""
    print("\n" + "=" * 80)
    print("Example 4: RAG-Enhanced Image Generation")
    print("=" * 80)

    # Initialize Chroma
    client = chromadb.Client()
    collection = client.create_collection(name="art_inspiration")

    # Add art descriptions
    collection.add(
        documents=[
            "Impressionist paintings feature visible brush strokes and emphasis on light and color",
            "Gothic architecture has pointed arches, ribbed vaults, and flying buttresses",
            "Cyberpunk aesthetics include neon lights, rain-soaked streets, and technological dystopia",
            "Japanese zen gardens emphasize simplicity, asymmetry, and natural elements",
            "Art Nouveau is characterized by organic flowing lines and nature-inspired motifs",
        ],
        ids=[f"art{i}" for i in range(1, 6)],
        metadatas=[
            {"period": "impressionism"},
            {"period": "gothic"},
            {"period": "modern"},
            {"period": "traditional"},
            {"period": "art_nouveau"},
        ]
    )

    # Create image generator
    img_gen = OpenAIImageGenerativeFunction(model="dall-e-3")

    # Generate image using RAG context
    print("\nGenerating image with context from collection...")
    result = collection.generate_image(
        prompt="A beautiful garden",
        image_generative_function=img_gen,
        style="art_nouveau",
        mood="serene",
        n_results=2,
        use_context=True
    )

    print(f"\nOriginal prompt: A beautiful garden")
    print(f"Enhanced prompt: {result['enhanced_prompt']}")
    print(f"Style: {result['style']}")
    print(f"Mood: {result['mood']}")
    print(f"Source documents: {result['source_documents']}")
    print(f"Image URL: {result['image'][:60]}...")

    client.delete_collection(name="art_inspiration")


def example_5_image_variations():
    """Example 5: Generate multiple variations of an image."""
    print("\n" + "=" * 80)
    print("Example 5: Image Variations")
    print("=" * 80)

    img_gen = OpenAIImageGenerativeFunction(model="dall-e-2")  # DALL-E 2 supports n>1

    print("\nGenerating 3 variations of 'A magical crystal cave'...")

    variations = img_gen.generate_variations(
        prompt="A magical crystal cave",
        n=3,
        style="fantasy",
        mood="mysterious"
    )

    for i, img_url in enumerate(variations, 1):
        print(f"  Variation {i}: {img_url[:60]}...")


def example_6_stable_diffusion():
    """Example 6: Using Stable Diffusion for more control."""
    print("\n" + "=" * 80)
    print("Example 6: Stable Diffusion with Advanced Controls")
    print("=" * 80)

    # Note: Requires STABILITY_API_KEY environment variable
    try:
        sd_gen = StableDiffusionGenerativeFunction(
            api_provider="stability",  # or "replicate" or "local"
            model="stable-diffusion-xl-1024-v1-0",
            width=1024,
            height=1024,
            steps=30,
            guidance_scale=7.5
        )

        print("\nGenerating with Stable Diffusion...")

        image_data = sd_gen(
            prompt="A majestic dragon flying over mountains",
            style="fantasy",
            mood="epic",
            steps=50,  # More steps = higher quality
            guidance_scale=8.5  # Higher = more adherence to prompt
        )

        print(f"Generated image (base64 data): {len(image_data)} characters")

    except ValueError as e:
        print(f"\nSkipping Stable Diffusion example: {e}")
        print("Set STABILITY_API_KEY environment variable to use Stability AI")


def example_7_image_transformation():
    """Example 7: Transform existing images (image-to-image)."""
    print("\n" + "=" * 80)
    print("Example 7: Image Transformation")
    print("=" * 80)

    img_gen = OpenAIImageGenerativeFunction(model="dall-e-2")

    print("\nNote: This example requires an existing image file.")
    print("Skipping actual transformation (requires image file)")

    # Example code (requires actual image file):
    # result = collection.transform_image(
    #     image_input="path/to/your/photo.jpg",
    #     transformation_prompt="Make it look like a Van Gogh painting",
    #     image_generative_function=img_gen,
    #     style="impressionist",
    #     mood="artistic",
    #     strength=0.8  # High strength for significant transformation
    # )
    # print(f"Transformed image: {result['image']}")


def example_8_style_and_mood_discovery():
    """Example 8: Discover available styles and moods."""
    print("\n" + "=" * 80)
    print("Example 8: Available Styles and Moods")
    print("=" * 80)

    # Get all available styles
    styles = get_available_styles()
    print(f"\n📐 Available Styles ({len(styles)} total):")
    for style in sorted(styles):
        print(f"  - {style}")

    # Get all available moods
    moods = get_available_moods()
    print(f"\n🎭 Available Moods ({len(moods)} total):")
    for mood in sorted(moods):
        print(f"  - {mood}")

    # Get style details
    print("\n🔍 Style Details Example:")
    img_gen = OpenAIImageGenerativeFunction(model="dall-e-3")

    print(f"\nDALL-E Predefined Styles: {img_gen.get_available_styles()[:5]}...")
    print(f"DALL-E Predefined Moods: {img_gen.get_available_moods()[:5]}...")


def example_9_combined_style_mood():
    """Example 9: Combining multiple styles and moods."""
    print("\n" + "=" * 80)
    print("Example 9: Advanced Style/Mood Combinations")
    print("=" * 80)

    img_gen = OpenAIImageGenerativeFunction(model="dall-e-3")

    # Different combinations
    combinations = [
        ("A bustling marketplace", "watercolor", "cheerful"),
        ("A lone lighthouse", "cinematic", "melancholic"),
        ("A space station", "3d render", "futuristic"),
        ("A mysterious portal", "surreal", "mysterious"),
    ]

    print("\nGenerating images with different style/mood combinations:\n")

    for prompt, style, mood in combinations:
        image_url = img_gen(prompt=prompt, style=style, mood=mood)
        print(f"  {prompt}")
        print(f"    Style: {style}, Mood: {mood}")
        print(f"    -> {image_url[:50]}...")
        print()


def example_10_quality_and_size_control():
    """Example 10: Control image quality and size."""
    print("\n" + "=" * 80)
    print("Example 10: Quality and Size Control")
    print("=" * 80)

    print("\nDALL-E 3 Quality Comparison:")

    # Standard quality
    img_gen_standard = OpenAIImageGenerativeFunction(
        model="dall-e-3",
        size="1024x1024",
        quality="standard"
    )

    standard_url = img_gen_standard(
        prompt="A detailed architectural rendering",
        style="photorealistic",
        mood="professional"
    )
    print(f"  Standard quality: {standard_url[:60]}...")

    # HD quality (costs more)
    img_gen_hd = OpenAIImageGenerativeFunction(
        model="dall-e-3",
        size="1024x1024",
        quality="hd"
    )

    hd_url = img_gen_hd(
        prompt="A detailed architectural rendering",
        style="photorealistic",
        mood="professional"
    )
    print(f"  HD quality: {hd_url[:60]}...")

    # Different sizes
    print("\nDALL-E 3 Size Options:")
    sizes = ["1024x1024", "1792x1024", "1024x1792"]

    for size in sizes:
        img_gen_size = OpenAIImageGenerativeFunction(
            model="dall-e-3",
            size=size
        )
        url = img_gen_size(prompt="A panoramic landscape", style="photorealistic")
        print(f"  {size}: {url[:50]}...")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("CHROMA IMAGE GENERATION FEATURES EXAMPLES")
    print("*" * 80)

    examples = [
        ("Basic Image Generation", example_1_basic_image_generation),
        ("Style Variations", example_2_style_variations),
        ("Mood Control", example_3_mood_control),
        ("RAG-Enhanced Generation", example_4_rag_enhanced_image_generation),
        ("Image Variations", example_5_image_variations),
        ("Stable Diffusion", example_6_stable_diffusion),
        ("Image Transformation", example_7_image_transformation),
        ("Style/Mood Discovery", example_8_style_and_mood_discovery),
        ("Combined Style/Mood", example_9_combined_style_mood),
        ("Quality/Size Control", example_10_quality_and_size_control),
    ]

    for name, example_fn in examples:
        try:
            example_fn()
        except Exception as e:
            print(f"\n❌ Example '{name}' failed: {e}")
            print("This might be due to missing API keys or dependencies.")
            continue

    print("\n" + "*" * 80)
    print("Examples completed!")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
