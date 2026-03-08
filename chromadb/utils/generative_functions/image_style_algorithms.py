"""
Image styling and mood manipulation algorithms.

This module provides advanced algorithms for controlling the style and mood
of generated or transformed images using various techniques.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class StyleCategory(Enum):
    """Categories of visual styles."""
    ARTISTIC = "artistic"
    PHOTOGRAPHIC = "photographic"
    ILLUSTRATION = "illustration"
    ABSTRACT = "abstract"
    PERIOD = "period"
    CULTURAL = "cultural"
    GENRE = "genre"


class MoodCategory(Enum):
    """Categories of emotional moods."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    ENERGETIC = "energetic"
    CALM = "calm"
    MYSTERIOUS = "mysterious"


@dataclass
class StyleProfile:
    """
    Complete style profile for image generation.

    Combines multiple aspects of visual style including art movement,
    technique, color palette, and composition.
    """

    name: str
    category: StyleCategory
    keywords: List[str]
    color_palette: Optional[str] = None
    lighting: Optional[str] = None
    texture: Optional[str] = None
    composition: Optional[str] = None
    reference_artists: Optional[List[str]] = None

    def to_prompt(self) -> str:
        """Convert style profile to prompt string."""
        parts = [self.name] + self.keywords

        if self.color_palette:
            parts.append(self.color_palette)
        if self.lighting:
            parts.append(self.lighting)
        if self.texture:
            parts.append(self.texture)
        if self.composition:
            parts.append(self.composition)
        if self.reference_artists:
            parts.append(f"in the style of {', '.join(self.reference_artists)}")

        return ", ".join(parts)


@dataclass
class MoodProfile:
    """
    Complete mood profile for atmospheric control.

    Defines the emotional atmosphere, lighting conditions, weather,
    and other environmental factors.
    """

    name: str
    category: MoodCategory
    keywords: List[str]
    lighting_condition: Optional[str] = None
    weather: Optional[str] = None
    time_of_day: Optional[str] = None
    color_temperature: Optional[str] = None
    atmosphere: Optional[str] = None

    def to_prompt(self) -> str:
        """Convert mood profile to prompt string."""
        parts = [self.name] + self.keywords

        if self.lighting_condition:
            parts.append(self.lighting_condition)
        if self.weather:
            parts.append(self.weather)
        if self.time_of_day:
            parts.append(self.time_of_day)
        if self.color_temperature:
            parts.append(self.color_temperature)
        if self.atmosphere:
            parts.append(self.atmosphere)

        return ", ".join(parts)


class StyleLibrary:
    """Comprehensive library of predefined styles."""

    STYLES: Dict[str, StyleProfile] = {
        # Artistic styles
        "impressionist": StyleProfile(
            name="impressionist",
            category=StyleCategory.ARTISTIC,
            keywords=["impressionist painting", "visible brush strokes", "light and color"],
            color_palette="vibrant colors",
            lighting="natural outdoor lighting",
            texture="thick paint texture",
            reference_artists=["Claude Monet", "Pierre-Auguste Renoir"],
        ),
        "cubist": StyleProfile(
            name="cubist",
            category=StyleCategory.ARTISTIC,
            keywords=["cubist art", "geometric shapes", "multiple perspectives"],
            color_palette="muted earth tones",
            composition="fragmented, abstract composition",
            reference_artists=["Pablo Picasso", "Georges Braque"],
        ),
        "surrealist": StyleProfile(
            name="surrealist",
            category=StyleCategory.ARTISTIC,
            keywords=["surrealist art", "dreamlike", "impossible scenes"],
            composition="unexpected juxtapositions",
            reference_artists=["Salvador Dalí", "René Magritte"],
        ),
        "art_nouveau": StyleProfile(
            name="art_nouveau",
            category=StyleCategory.ARTISTIC,
            keywords=["art nouveau", "organic forms", "decorative", "flowing lines"],
            color_palette="rich, harmonious colors",
            composition="ornate, nature-inspired patterns",
            reference_artists=["Alphonse Mucha", "Gustav Klimt"],
        ),

        # Photographic styles
        "cinematic": StyleProfile(
            name="cinematic",
            category=StyleCategory.PHOTOGRAPHIC,
            keywords=["cinematic", "film still", "anamorphic lens"],
            lighting="dramatic three-point lighting",
            composition="rule of thirds, depth of field",
            color_palette="color graded, cinematic color",
        ),
        "portrait_photography": StyleProfile(
            name="portrait_photography",
            category=StyleCategory.PHOTOGRAPHIC,
            keywords=["portrait photography", "professional", "studio quality"],
            lighting="soft studio lighting",
            composition="bokeh background, shallow depth of field",
            texture="sharp focus on subject",
        ),
        "street_photography": StyleProfile(
            name="street_photography",
            category=StyleCategory.PHOTOGRAPHIC,
            keywords=["street photography", "candid", "documentary style"],
            lighting="natural light",
            composition="decisive moment, authentic",
        ),

        # Illustration styles
        "anime": StyleProfile(
            name="anime",
            category=StyleCategory.ILLUSTRATION,
            keywords=["anime style", "cel shaded", "vibrant colors"],
            lighting="anime lighting",
            composition="dynamic composition",
            texture="clean lines, digital painting",
        ),
        "comic_book": StyleProfile(
            name="comic_book",
            category=StyleCategory.ILLUSTRATION,
            keywords=["comic book style", "bold lines", "dynamic"],
            color_palette="vibrant, saturated colors",
            texture="halftone dots, ink",
            composition="action-packed, dramatic angles",
        ),
        "watercolor_illustration": StyleProfile(
            name="watercolor_illustration",
            category=StyleCategory.ILLUSTRATION,
            keywords=["watercolor illustration", "soft edges", "flowing"],
            color_palette="delicate, translucent colors",
            texture="paper texture, water stains",
        ),

        # Period styles
        "victorian": StyleProfile(
            name="victorian",
            category=StyleCategory.PERIOD,
            keywords=["Victorian era", "ornate", "detailed"],
            color_palette="rich, deep colors",
            composition="elaborate, symmetrical",
            texture="vintage, aged",
        ),
        "art_deco": StyleProfile(
            name="art_deco",
            category=StyleCategory.PERIOD,
            keywords=["art deco", "geometric", "glamorous"],
            color_palette="gold, black, bold colors",
            composition="symmetrical, streamlined",
            texture="metallic, sleek",
        ),

        # Genre styles
        "cyberpunk": StyleProfile(
            name="cyberpunk",
            category=StyleCategory.GENRE,
            keywords=["cyberpunk", "neon lights", "futuristic", "dystopian"],
            lighting="neon glow, dramatic shadows",
            color_palette="neon pink, cyan, purple",
            atmosphere="rain, foggy, nighttime",
        ),
        "steampunk": StyleProfile(
            name="steampunk",
            category=StyleCategory.GENRE,
            keywords=["steampunk", "Victorian", "mechanical", "brass and copper"],
            color_palette="bronze, brass, sepia tones",
            texture="industrial, gears and machinery",
            atmosphere="steam, industrial",
        ),
    }

    @classmethod
    def get_style(cls, name: str) -> Optional[StyleProfile]:
        """Get a style profile by name."""
        return cls.STYLES.get(name.lower().replace(" ", "_"))

    @classmethod
    def list_styles(cls, category: Optional[StyleCategory] = None) -> List[str]:
        """List all available styles, optionally filtered by category."""
        if category:
            return [
                name for name, style in cls.STYLES.items()
                if style.category == category
            ]
        return list(cls.STYLES.keys())

    @classmethod
    def combine_styles(cls, styles: List[str], weights: Optional[List[float]] = None) -> str:
        """
        Combine multiple styles with optional weights.

        Args:
            styles: List of style names
            weights: Optional weights for each style (0-1)

        Returns:
            Combined prompt string
        """
        if weights is None:
            weights = [1.0 / len(styles)] * len(styles)

        prompts = []
        for style_name, weight in zip(styles, weights):
            style = cls.get_style(style_name)
            if style:
                if weight < 1.0:
                    prompts.append(f"({style.to_prompt()}:{weight:.2f})")
                else:
                    prompts.append(style.to_prompt())

        return ", ".join(prompts)


class MoodLibrary:
    """Comprehensive library of predefined moods."""

    MOODS: Dict[str, MoodProfile] = {
        # Positive moods
        "cheerful": MoodProfile(
            name="cheerful",
            category=MoodCategory.POSITIVE,
            keywords=["cheerful", "happy", "uplifting", "bright"],
            lighting_condition="bright, warm lighting",
            weather="sunny, clear skies",
            time_of_day="morning",
            color_temperature="warm tones",
            atmosphere="optimistic, energetic",
        ),
        "serene": MoodProfile(
            name="serene",
            category=MoodCategory.CALM,
            keywords=["serene", "peaceful", "tranquil", "calm"],
            lighting_condition="soft, diffused light",
            weather="clear, gentle breeze",
            time_of_day="golden hour",
            color_temperature="neutral, balanced",
            atmosphere="quiet, meditative",
        ),
        "romantic": MoodProfile(
            name="romantic",
            category=MoodCategory.POSITIVE,
            keywords=["romantic", "dreamy", "intimate", "soft"],
            lighting_condition="soft golden hour light",
            weather="misty, ethereal",
            time_of_day="sunset",
            color_temperature="warm, rosy tones",
            atmosphere="tender, nostalgic",
        ),

        # Negative moods
        "melancholic": MoodProfile(
            name="melancholic",
            category=MoodCategory.NEGATIVE,
            keywords=["melancholic", "somber", "wistful", "pensive"],
            lighting_condition="overcast, muted light",
            weather="rainy, cloudy",
            time_of_day="dusk",
            color_temperature="cool, desaturated",
            atmosphere="reflective, lonely",
        ),
        "ominous": MoodProfile(
            name="ominous",
            category=MoodCategory.NEGATIVE,
            keywords=["ominous", "foreboding", "threatening", "dark"],
            lighting_condition="low key, dramatic shadows",
            weather="stormy, dark clouds",
            time_of_day="night",
            color_temperature="cold, dark tones",
            atmosphere="tense, unsettling",
        ),

        # Mysterious moods
        "mysterious": MoodProfile(
            name="mysterious",
            category=MoodCategory.MYSTERIOUS,
            keywords=["mysterious", "enigmatic", "secretive", "intriguing"],
            lighting_condition="chiaroscuro, hidden in shadows",
            weather="foggy, obscured",
            time_of_day="twilight",
            color_temperature="cool, mysterious tones",
            atmosphere="suspenseful, alluring",
        ),

        # Energetic moods
        "dynamic": MoodProfile(
            name="dynamic",
            category=MoodCategory.ENERGETIC,
            keywords=["dynamic", "energetic", "powerful", "intense"],
            lighting_condition="high contrast, dramatic",
            weather="turbulent, windy",
            atmosphere="action-packed, thrilling",
        ),
        "epic": MoodProfile(
            name="epic",
            category=MoodCategory.ENERGETIC,
            keywords=["epic", "grand", "majestic", "awe-inspiring"],
            lighting_condition="dramatic god rays, volumetric",
            composition="wide angle, vast scale",
            atmosphere="heroic, monumental",
        ),

        # Calm moods
        "meditative": MoodProfile(
            name="meditative",
            category=MoodCategory.CALM,
            keywords=["meditative", "zen", "contemplative", "still"],
            lighting_condition="soft, even lighting",
            weather="calm, still air",
            time_of_day="early morning",
            color_temperature="neutral, balanced",
            atmosphere="mindful, centered",
        ),

        # Whimsical
        "whimsical": MoodProfile(
            name="whimsical",
            category=MoodCategory.POSITIVE,
            keywords=["whimsical", "playful", "fantastical", "imaginative"],
            lighting_condition="magical, sparkly light",
            color_temperature="vibrant, saturated colors",
            atmosphere="lighthearted, dreamlike",
        ),
    }

    @classmethod
    def get_mood(cls, name: str) -> Optional[MoodProfile]:
        """Get a mood profile by name."""
        return cls.MOODS.get(name.lower().replace(" ", "_"))

    @classmethod
    def list_moods(cls, category: Optional[MoodCategory] = None) -> List[str]:
        """List all available moods, optionally filtered by category."""
        if category:
            return [
                name for name, mood in cls.MOODS.items()
                if mood.category == category
            ]
        return list(cls.MOODS.keys())


class StyleMoodCombiner:
    """Utilities for combining styles and moods effectively."""

    # Compatibility matrix (some styles/moods work better together)
    COMPATIBILITY_SCORES: Dict[Tuple[str, str], float] = {
        ("impressionist", "cheerful"): 1.0,
        ("impressionist", "serene"): 0.9,
        ("cyberpunk", "ominous"): 1.0,
        ("cyberpunk", "dynamic"): 0.95,
        ("watercolor_illustration", "serene"): 1.0,
        ("watercolor_illustration", "whimsical"): 0.95,
        ("cinematic", "epic"): 1.0,
        ("cinematic", "dramatic"): 0.95,
        ("surrealist", "mysterious"): 0.9,
        ("surrealist", "whimsical"): 0.85,
    }

    @classmethod
    def get_compatibility(cls, style: str, mood: str) -> float:
        """
        Get compatibility score between style and mood.

        Returns:
            Score from 0-1 (higher = more compatible)
        """
        key = (style.lower(), mood.lower())
        return cls.COMPATIBILITY_SCORES.get(key, 0.5)  # Default neutral

    @classmethod
    def suggest_mood_for_style(cls, style: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Suggest compatible moods for a given style.

        Args:
            style: Style name
            top_n: Number of suggestions to return

        Returns:
            List of (mood_name, compatibility_score) tuples
        """
        style_lower = style.lower()
        scores = []

        for mood_name in MoodLibrary.list_moods():
            score = cls.get_compatibility(style_lower, mood_name)
            scores.append((mood_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    @classmethod
    def suggest_style_for_mood(cls, mood: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Suggest compatible styles for a given mood.

        Args:
            mood: Mood name
            top_n: Number of suggestions to return

        Returns:
            List of (style_name, compatibility_score) tuples
        """
        mood_lower = mood.lower()
        scores = []

        for style_name in StyleLibrary.list_styles():
            score = cls.get_compatibility(style_name, mood_lower)
            scores.append((style_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    @classmethod
    def create_enhanced_prompt(
        cls,
        base_prompt: str,
        style: Optional[str] = None,
        mood: Optional[str] = None,
        enhance_quality: bool = True,
    ) -> str:
        """
        Create an enhanced prompt combining base, style, and mood.

        Args:
            base_prompt: Base description
            style: Style name
            mood: Mood name
            enhance_quality: Add quality boosters

        Returns:
            Enhanced prompt string
        """
        parts = [base_prompt]

        if style:
            style_profile = StyleLibrary.get_style(style)
            if style_profile:
                parts.append(style_profile.to_prompt())

        if mood:
            mood_profile = MoodLibrary.get_mood(mood)
            if mood_profile:
                parts.append(mood_profile.to_prompt())

        if enhance_quality:
            parts.append("high quality, detailed, masterpiece")

        return ", ".join(parts)


# Helper functions for easy access
def get_available_styles() -> List[str]:
    """Get list of all available styles."""
    return StyleLibrary.list_styles()


def get_available_moods() -> List[str]:
    """Get list of all available moods."""
    return MoodLibrary.list_moods()


def enhance_prompt_with_style_mood(
    prompt: str,
    style: Optional[str] = None,
    mood: Optional[str] = None,
) -> str:
    """
    Quick helper to enhance a prompt with style and mood.

    Args:
        prompt: Base prompt
        style: Style name
        mood: Mood name

    Returns:
        Enhanced prompt
    """
    return StyleMoodCombiner.create_enhanced_prompt(prompt, style, mood)
