"""
LLM-based agent for Han et al. (2024) reproduction experiments.

Follows Han's multi-turn conversation approach with tutorial phase.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

from src.infra.llm.client import LLMClient
from src.infra.llm.message import Message, MessageRole
from src.infra.llm.models import Models

from ..config import Action, PayoffMatrix
from ..prompts.templates import (
    build_tutorial_messages,
    build_round_prompt,
    compute_last_round_info,
)
from ..model_config import get_model_extra_body, get_model_reasoning_effort


class Decision(BaseModel):
    action: Literal["C", "D"] = Field(
        description="The action to take: 'C' for Cooperate or 'D' for Defect"
    )


@dataclass
class RoundMemory:
    """Memory of a full round in the network."""

    round_num: int
    my_action: Action
    neighbor_actions: Dict[int, Action]
    my_total_payoff: int
    last_round_info: Dict[str, Any]  # Full info for prompt generation


class LLMAgent:
    """
    LLM-based agent for Iterated Prisoner's Dilemma.

    Follows Han et al. (2024) conversation structure:
    1. Tutorial phase establishes rules
    2. Each round builds on conversation history with last-round feedback
    """

    def __init__(
        self,
        agent_id: int = 0,
        name: str = "LLM",
        llm_client: Optional[LLMClient] = None,
        model: Models = Models.GPT_3_5_TURBO,  # Han et al. default
        temperature: float = 1.0,  # Han et al. default (OpenAI default)
        memory_window: int = 10,
        payoff_matrix: Optional[PayoffMatrix] = None,
        disable_cache: bool = False,
        with_reasoning: bool = False,  # Enable reasoning/extended thinking mode
    ):
        self.agent_id = agent_id
        self.name = name
        self.llm_client = llm_client or LLMClient()
        self.model = model
        self.temperature = temperature
        self.memory_window = memory_window
        self.payoff_matrix = payoff_matrix or PayoffMatrix()
        self.disable_cache = disable_cache
        self.with_reasoning = with_reasoning

        # Memory: list of rounds (most recent last)
        self._memory: List[RoundMemory] = []

        # Conversation history for multi-turn dialog
        self._conversation: List[Message] = []
        self._tutorial_done = False

    def _run_tutorial(self, max_rounds: int, is_well_mixed: bool):
        """Run the tutorial conversation to establish game rules."""
        tutorial_msgs = build_tutorial_messages(
            payoff_matrix=self.payoff_matrix,
            max_rounds=max_rounds,
            is_well_mixed=is_well_mixed,
        )

        # Han's tutorial has assistant responses between user messages
        # We'll simulate this by adding placeholder assistant responses
        for msg in tutorial_msgs:
            if msg["role"] == "system":
                role = MessageRole.SYSTEM
            elif msg["role"] == "user":
                role = MessageRole.USER
            else:
                role = MessageRole.ASSISTANT

            self._conversation.append(Message(role=role, content=msg["content"]))

            # Add a brief assistant acknowledgment after each message (system or user)
            # Han's dialog shows assistant responses after EVERY message including the system message
            if role in (MessageRole.SYSTEM, MessageRole.USER):
                self._conversation.append(
                    Message(role=MessageRole.ASSISTANT, content="Yes, I understand.")
                )

        self._tutorial_done = True

    def decide(
        self,
        round_num: int,
        max_rounds: int,
        payoff_matrix: PayoffMatrix,
        neighbor_ids: List[int],
        neighbor_history: Dict[int, List[Action]],  # Not used in Han's format
        neighbor_points: Dict[int, int],  # Not used in Han's format
        replication_id: Optional[int] = None,
        is_well_mixed: bool = False,
        last_round_info: Optional[Dict[str, Any]] = None,  # NEW: computed by env
    ) -> Action:
        """
        Make a decision using LLM.

        Args:
            last_round_info: Detailed info about last round in Han's format
                             (computed by environment using compute_last_round_info)
        """
        # Run tutorial on first decision
        if not self._tutorial_done:
            self._run_tutorial(max_rounds, is_well_mixed)

        # Build round prompt
        round_prompt = build_round_prompt(round_num, last_round_info)

        # Add to conversation
        self._conversation.append(Message(role=MessageRole.USER, content=round_prompt))

        log_tags = [
            f"agent_id:{self.agent_id}",
            f"round:{round_num}",
            f"well_mixed:{is_well_mixed}",
        ]
        if replication_id is not None:
            log_tags.append(f"replication:{replication_id}")

        # Get model-specific configuration
        extra_body = get_model_extra_body(self.model, self.disable_cache, self.with_reasoning)
        reasoning_effort = get_model_reasoning_effort(self.model, self.with_reasoning)

        # For Claude reasoning mode, temperature must be exactly 1.0
        temp = self.temperature
        if self.with_reasoning and self.model == Models.CLAUDE_SONNET_4_5:
            temp = 1.0

        response = self.llm_client.generate(
            model=self.model,
            messages=self._conversation,
            temperature=temp,
            response_format=Decision,
            reasoning_effort=reasoning_effort,
            log_tags=log_tags,
            extra_body=extra_body,
        )

        # Add assistant response to conversation
        action_str = response.content.action
        self._conversation.append(
            Message(role=MessageRole.ASSISTANT, content=action_str)
        )

        return Action.COOPERATE if action_str.upper() == "C" else Action.DEFECT

    def add_round_result(
        self,
        my_action: Action,
        neighbor_actions: Dict[int, Action],
        my_total_payoff: int,
        round_num: int,
        last_round_info: Dict[str, Any],
    ) -> None:
        """Add a round's results to memory."""
        memory = RoundMemory(
            round_num=round_num,
            my_action=my_action,
            neighbor_actions=neighbor_actions,
            my_total_payoff=my_total_payoff,
            last_round_info=last_round_info,
        )
        self._memory.append(memory)

    @property
    def total_payoff(self) -> int:
        """Calculate total cumulative payoff."""
        return sum(m.my_total_payoff for m in self._memory)

    def reset(self):
        """Reset agent state for a new game."""
        self._memory = []
        self._conversation = []
        self._tutorial_done = False
