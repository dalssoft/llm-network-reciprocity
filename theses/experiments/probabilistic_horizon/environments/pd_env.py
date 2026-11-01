"""
Networked Prisoner's Dilemma environment with stochastic termination.

Modified from modern_models version: replaces fixed round loop with
probabilistic continuation based on delta parameter.
"""

import random
from typing import Dict, List, Optional, Any, Callable
from ..config import Action, PayoffMatrix, ProbHorizonConfig
from ..agents.agent import LLMAgent
from ..prompts.templates import compute_last_round_info
from ...common.network import NetworkTopology, LatticeNetwork, WellMixedNetwork


class NetworkedPD:
    """
    Manages a population of agents playing PD on a network.

    Uses stochastic termination: after min_rounds, each round continues
    with probability continuation_delta. Capped at max_rounds.
    """

    def __init__(
        self,
        agents: List[LLMAgent],
        network: NetworkTopology,
        config: ProbHorizonConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.agents = {a.agent_id: a for a in agents}
        self.network = network
        self.config = config
        self.history: List[Dict] = []
        self.current_round = 0
        self.progress_callback = progress_callback

        # Store last round's actions for computing feedback
        self._last_round_actions: Dict[int, Action] = {}
        # Store network structure for each round
        self._last_round_neighbors: Dict[int, List[int]] = {}

    def _get_network_neighbors(self) -> Dict[int, List[int]]:
        """Get current network structure as dict."""
        return {
            agent_id: self.network.get_neighbors(agent_id)
            for agent_id in self.agents.keys()
        }

    def run_round(self):
        """Run a single round of the simulation."""
        self.current_round += 1

        # If well-mixed, shuffle the network each round
        is_well_mixed = isinstance(self.network, WellMixedNetwork)
        if is_well_mixed:
            self.network.randomize_matching()

        # Get current network structure
        current_neighbors = self._get_network_neighbors()

        # 1. Collect decisions from all agents
        round_decisions: Dict[int, Action] = {}

        for agent_id, agent in self.agents.items():
            neighbor_ids = current_neighbors[agent_id]

            # Compute last round info for prompt (Han's format)
            last_round_info = None
            if self.current_round > 1 and self._last_round_actions:
                my_last_action = self._last_round_actions.get(agent_id)
                if my_last_action is not None:
                    # Get last round's neighbors (might differ in well-mixed)
                    last_neighbors = self._last_round_neighbors.get(agent_id, [])
                    last_neighbor_actions = {
                        n_id: self._last_round_actions[n_id]
                        for n_id in last_neighbors
                        if n_id in self._last_round_actions
                    }

                    last_round_info = compute_last_round_info(
                        agent_id=agent_id,
                        my_action=my_last_action,
                        neighbor_actions=last_neighbor_actions,
                        payoff_matrix=self.config.payoff_matrix,
                        all_agent_actions=self._last_round_actions,
                        network_neighbors=self._last_round_neighbors,
                    )

            # Agent makes decision (no max_rounds passed — agent doesn't know)
            decision = agent.decide(
                round_num=self.current_round,
                payoff_matrix=self.config.payoff_matrix,
                neighbor_ids=neighbor_ids,
                neighbor_history={},  # Not used in Han's format
                neighbor_points={},  # Not used in Han's format
                replication_id=self.config.seed,
                is_well_mixed=is_well_mixed,
                last_round_info=last_round_info,
            )
            round_decisions[agent_id] = decision

        # 2. Calculate payoffs and update memories
        round_results = []
        for agent_id, decision in round_decisions.items():
            neighbors = current_neighbors[agent_id]
            neighbor_actions = {n_id: round_decisions[n_id] for n_id in neighbors}

            # Compute this round's info (will be used next round)
            round_info = compute_last_round_info(
                agent_id=agent_id,
                my_action=decision,
                neighbor_actions=neighbor_actions,
                payoff_matrix=self.config.payoff_matrix,
                all_agent_actions=round_decisions,
                network_neighbors=current_neighbors,
            )

            total_payoff = round_info["my_total"]

            # Update agent memory
            self.agents[agent_id].add_round_result(
                my_action=decision,
                neighbor_actions=neighbor_actions,
                my_total_payoff=total_payoff,
                round_num=self.current_round,
                last_round_info=round_info,
            )

            round_results.append(
                {
                    "agent_id": agent_id,
                    "action": decision,
                    "payoff": total_payoff,
                    "neighbors": neighbors,
                }
            )

        # Store for next round's feedback
        self._last_round_actions = round_decisions.copy()
        self._last_round_neighbors = current_neighbors.copy()

        self.history.append({"round": self.current_round, "results": round_results})

    def run_simulation(self):
        """
        Run the simulation with stochastic termination.

        After min_rounds, each round continues with probability
        continuation_delta. Always stops at max_rounds.
        """
        rng = random.Random(self.config.seed)

        while self.current_round < self.config.max_rounds:
            self.run_round()

            if self.progress_callback:
                self.progress_callback(self.current_round, self.config.max_rounds)

            # Check stochastic termination after min_rounds
            if self.current_round >= self.config.min_rounds:
                if rng.random() > self.config.continuation_delta:
                    break

        return self.history
