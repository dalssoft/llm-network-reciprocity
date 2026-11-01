"""
Model-specific configuration helper for modern models experiment.

Handles special configurations like reasoning effort and extended thinking.
"""

from typing import Dict, Any, Optional, Tuple
from src.infra.llm.models import Models, ReasoningEffort


# Define model configurations: (model, with_reasoning)
# This allows testing same model with and without reasoning
MODEL_CONFIGS = [
    # DeepSeek-V3.2 - standard only
    (Models.DEEPSEEK_V3_2, False),
    # Gemini 3 Flash - with and without reasoning
    (Models.GEMINI_3_FLASH, False),  # Standard mode
    (Models.GEMINI_3_FLASH, True),  # With reasoning HIGH
    # Gemini 2.5 Flash Lite - with and without reasoning (cheap/fast)
    (Models.GEMINI_2_5_FLASH_LITE, False),  # Standard mode
    (Models.GEMINI_2_5_FLASH_LITE, True),  # With reasoning HIGH
    # GPT-5-mini - with and without reasoning
    (Models.GPT_5_MINI, False),  # Standard mode
    (Models.GPT_5_MINI, True),  # With reasoning/thinking
    # GPT-OSS-120B - standard only (open source sample)
    (Models.GPT_OSS_120B, False),
]


def get_model_reasoning_effort(
    model: Models, with_reasoning: bool = False
) -> Optional[ReasoningEffort]:
    """
    Get the appropriate reasoning effort level for a model.

    Args:
        model: The model
        with_reasoning: Whether to enable reasoning mode

    Returns:
        ReasoningEffort level, or None for models without reasoning support
    """
    # Models that support reasoning_effort parameter via LiteLLM
    reasoning_models = [
        Models.GEMINI_3_FLASH,
        Models.GEMINI_2_5_FLASH_LITE,
        Models.GPT_5_MINI,
    ]
    if model in reasoning_models:
        return ReasoningEffort.HIGH if with_reasoning else None
    return None


def get_model_extra_body(
    model: Models, disable_cache: bool = False, with_reasoning: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Get model-specific extra_body configuration for LLM API calls.

    Args:
        model: The model to get configuration for
        disable_cache: Whether to disable caching
        with_reasoning: Whether to enable reasoning/thinking mode (unused - use reasoning_effort instead)

    Returns:
        Dictionary to pass as extra_body, or None if no special config needed
    """
    extra_body = {}

    # Handle cache settings
    if disable_cache:
        extra_body["cache"] = {"no-cache": True}

    # Note: Both Gemini and Claude use reasoning_effort parameter via LiteLLM
    # No extra_body configuration needed for reasoning

    return extra_body if extra_body else None


def get_config_description(model: Models, with_reasoning: bool) -> str:
    """Get human-readable description of model configuration."""

    base_names = {
        Models.DEEPSEEK_V3_2: "DeepSeek-V3.2",
        Models.GEMINI_3_FLASH: "Gemini 3 Flash",
        Models.GEMINI_2_5_FLASH_LITE: "Gemini 2.5 Flash Lite",
        Models.GPT_5_MINI: "GPT-5-mini",
        Models.GPT_OSS_120B: "GPT-OSS-120B",
    }

    base = base_names.get(model, model.model_name)

    if not with_reasoning:
        return f"{base} (standard)"

    return f"{base} (reasoning: HIGH)"


def get_config_id(model: Models, with_reasoning: bool) -> str:
    """Get a short identifier for this configuration (for file naming)."""
    name = model.model_name.replace("-", "_").replace(".", "_")
    suffix = "_reasoning" if with_reasoning else ""
    return f"{name}{suffix}"


def get_model_description(model: Models) -> str:
    """Get human-readable description of a model (without reasoning suffix)."""
    base_names = {
        Models.DEEPSEEK_V3_2: "DeepSeek-V3.2",
        Models.GEMINI_3_FLASH: "Gemini 3 Flash",
        Models.GEMINI_2_5_FLASH_LITE: "Gemini 2.5 Flash Lite",
        Models.GPT_5_MINI: "GPT-5-mini",
        Models.GPT_OSS_120B: "GPT-OSS-120B",
    }
    return base_names.get(model, model.model_name)


# Convenience lists
MODERN_MODELS = [model for model, _ in MODEL_CONFIGS]
MODERN_CONFIGS = MODEL_CONFIGS  # List of (model, with_reasoning) tuples
