"""
Configuration for Han et al. (2024) reproduction experiments.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

from src.infra.llm.models import Models


class Action(Enum):
    """Prisoner's Dilemma actions."""

    COOPERATE = "C"
    DEFECT = "D"


@dataclass
class PayoffMatrix:
    """
    Payoff matrix for Prisoner's Dilemma.
    Standard PD requires: T > R > P > S and 2R > T + S
    """

    T: int = 5  # Temptation
    R: int = 3  # Reward
    P: int = 1  # Punishment
    S: int = 0  # Sucker

    def __post_init__(self):
        """Validate PD constraints."""
        if not (self.T > self.R > self.P > self.S):
            raise ValueError(
                f"Invalid PD payoffs: must satisfy T > R > P > S. "
                f"Got T={self.T}, R={self.R}, P={self.P}, S={self.S}"
            )
        if not (2 * self.R > self.T + self.S):
            raise ValueError(
                f"Invalid PD payoffs: must satisfy 2R > T + S. "
                f"Got 2R={2*self.R}, T+S={self.T + self.S}"
            )

    @classmethod
    def from_bc_ratio(cls, bc_ratio: float, offset: int = 0):
        """
        Create payoff matrix from b/c ratio.
        Cost per neighbor interaction = 10.
        Benefit per neighbor interaction = 10 * bc_ratio.
        """
        b = 10 * bc_ratio
        c = 10

        return cls(
            T=int(b + offset), R=int(b - c + offset), P=offset, S=int(offset - c)
        )

    def get_payoffs(self, action1: Action, action2: Action) -> tuple[int, int]:
        """Get payoffs for both players."""
        if action1 == Action.COOPERATE:
            if action2 == Action.COOPERATE:
                return (self.R, self.R)
            else:
                return (self.S, self.T)
        else:
            if action2 == Action.COOPERATE:
                return (self.T, self.S)
            else:
                return (self.P, self.P)


@dataclass
class HanConfig:
    """Configuration for Han et al. (2024) reproduction."""

    # Game parameters
    num_rounds: int = 25
    payoff_matrix: PayoffMatrix = field(default_factory=PayoffMatrix)

    # Network parameters
    num_agents: int = 8
    k: int = 2  # Degree for ring network

    # Agent settings
    memory_window: int = 10  # Not used in Han's format (only last round shown)
    model: Models = Models.GPT_3_5_TURBO
    temperature: float = 1.0  # Paper doesn't specify; using OpenAI default 1.0

    # Simulation settings
    seed: Optional[int] = None
    replications: int = 5  # Han uses 5 replications


# Setting presets matching Han et al.
SETTING_1 = {
    "num_agents": 8,
    "num_rounds": 25,
    "k": 2,
    "bc_ratio": 6.0,
    "replications": 5,
}

SETTING_2 = {
    "num_agents": 25,
    "num_rounds": 15,
    "ks": [2, 4, 6],
    "bc_ratios": [2.0, 4.0, 6.0],
    "replications": 5,
}
