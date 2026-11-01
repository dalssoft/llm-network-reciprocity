"""
Configuration for probabilistic horizon experiment.

Tests whether a known endpoint is necessary for endgame collapse
by replacing fixed rounds with stochastic continuation probability.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.infra.llm.models import Models


class Action(Enum):
    """Prisoner's Dilemma actions."""

    COOPERATE = "C"
    DEFECT = "D"


class HorizonInfo(Enum):
    """Whether agents are told about the stochastic horizon."""

    INFORMED = "informed"
    UNINFORMED = "uninformed"


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
class ProbHorizonConfig:
    """Configuration for probabilistic horizon experiment."""

    # Game parameters
    payoff_matrix: PayoffMatrix = field(default_factory=PayoffMatrix)

    # Stochastic horizon parameters
    continuation_delta: float = 0.9  # Probability game continues each round
    min_rounds: int = 1  # Minimum rounds before stochastic termination
    max_rounds: int = 100  # Safety cap
    horizon_info: HorizonInfo = HorizonInfo.INFORMED

    # Network parameters
    num_agents: int = 25
    k: int = 2  # Degree for ring network

    # Agent settings
    model: Models = Models.GEMINI_3_FLASH
    temperature: float = 1.0
    with_reasoning: bool = True  # Default: Gemini 3 Flash (R)

    # Simulation settings
    seed: Optional[int] = None
    replications: int = 10
