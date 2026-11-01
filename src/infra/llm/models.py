from dataclasses import dataclass
from enum import Enum
from openai import NotGiven
from typing import Optional


@dataclass
class ModelCapabilities:
    """Metadata about model capabilities"""

    provider: str
    model_name: str
    supports_reasoning: bool = False
    supports_tools: bool = True
    supports_structured_output: bool = True


class Models(Enum):
    """Model enum with rich capability metadata"""

    GEMINI_2_5_FLASH_LITE = ModelCapabilities(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        supports_structured_output=True,
        supports_reasoning=True,  # Supports thinking mode
    )
    GEMINI_2_5_FLASH = ModelCapabilities(
        provider="gemini",
        model_name="gemini-2.5-flash",
    )
    GEMINI_2_5_PRO = ModelCapabilities(
        provider="gemini",
        model_name="gemini-2.5-pro",
    )
    GEMINI_3_FLASH = ModelCapabilities(
        provider="gemini",
        model_name="gemini-3-flash-preview",
        supports_reasoning=True,
    )
    DEEPSEEK_V3_2 = ModelCapabilities(
        provider="deepseek",
        model_name="DeepSeek-V3.2",
        supports_structured_output=False,
    )
    DEEPSEEK_V3_2_REASONER = ModelCapabilities(
        provider="deepseek",
        model_name="DeepSeek-V3.2-Reasoner",
        supports_structured_output=False,
    )
    O3_MINI = ModelCapabilities(
        provider="openai",
        model_name="o3-mini",
    )
    GPT_5_MINI = ModelCapabilities(
        provider="openai",
        model_name="gpt-5-mini",
        supports_reasoning=True,  # Supports thinking mode
    )
    GPT_5 = ModelCapabilities(
        provider="openai",
        model_name="gpt-5",
    )
    GPT_4_1 = ModelCapabilities(
        provider="openai",
        model_name="gpt-4.1",
    )
    GPT_4_0613 = ModelCapabilities(
        provider="openai",
        model_name="gpt-4-0613",
        supports_structured_output=False,
    )
    GPT_3_5_TURBO = ModelCapabilities(
        provider="openai",
        model_name="gpt-3.5-turbo",
        supports_structured_output=False,
    )
    # Successors for reproduction
    GPT_4O = ModelCapabilities(
        provider="openai",
        model_name="gpt-4o",
    )
    GPT_4O_MINI = ModelCapabilities(
        provider="openai",
        model_name="gpt-4o-mini",
    )
    CLAUDE_3_5_SONNET = ModelCapabilities(
        provider="anthropic",
        model_name="claude-3-5-sonnet",
    )
    CLAUDE_SONNET_4_5 = ModelCapabilities(
        provider="anthropic",
        model_name="claude-sonnet-4-5",
    )
    CLAUDE_SONNET_5 = ModelCapabilities(
        provider="anthropic",
        model_name="claude-sonnet-5",
        supports_reasoning=True,
    )
    GPT_5_4_MINI = ModelCapabilities(
        provider="openai",
        model_name="gpt-5.4-mini",
        supports_reasoning=True,
    )
    MISTRAL_SMALL_3_2 = ModelCapabilities(
        provider="mistral",
        model_name="mistral-small-3.2",
        supports_structured_output=False,
    )
    MIXTRAL_8X7B = ModelCapabilities(
        provider="mistral",
        model_name="mixtral-8x7b",
        supports_structured_output=False,
    )
    PERPLEXITY_SONAR_PRO = ModelCapabilities(
        provider="perplexity",
        model_name="pplx-sonar-pro",
    )
    GPT_OSS_120B = ModelCapabilities(
        provider="cerebras",
        model_name="gpt-oss-120b",
        supports_structured_output=False,
    )
    # Llama models via Cerebras - useful for replicating Fontana et al. (2024)
    LLAMA_3_1_8B = ModelCapabilities(
        provider="cerebras",
        model_name="llama3.1-8b",
        supports_structured_output=False,
    )
    LLAMA_3_3_70B = ModelCapabilities(
        provider="cerebras",
        model_name="llama-3.3-70b",
        supports_structured_output=False,
    )
    SABIA_3 = ModelCapabilities(
        provider="maritaca",
        model_name="openai/sabia-3",
    )

    @property
    def provider(self) -> str:
        return self.value.provider

    @property
    def model_name(self) -> str:
        """Full model identifier for LiteLLM"""
        return self.value.model_name

    @property
    def supports_reasoning(self) -> bool:
        return self.value.supports_reasoning

    @property
    def supports_tools(self) -> bool:
        return self.value.supports_tools

    @property
    def supports_structured_output(self) -> bool:
        return self.value.supports_structured_output

    @classmethod
    def from_model_name(cls, name: str) -> "Models":
        """Look up a model by its model_name string."""
        for model in cls:
            if model.model_name == name:
                return model
        raise ValueError(
            f"Unknown model: {name}. Available: {[m.model_name for m in cls]}"
        )


class ReasoningEffort(Enum):
    NONE = "disable"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NOT_SPECIFIED = "not_specified"

    @property
    def effort(self):
        if self == ReasoningEffort.NOT_SPECIFIED:
            return NotGiven()
        return self.value


@dataclass
class ModelConfig:
    """Configuration for a specific model attempt"""

    model: Models
    temperature: Optional[float] = None
    reasoning_effort: Optional[ReasoningEffort] = None
    max_tokens: Optional[int] = None
