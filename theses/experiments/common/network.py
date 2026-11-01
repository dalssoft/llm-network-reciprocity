"""
Network topology implementations for IPD simulations using NetworkX.

Supports:
- LatticeNetwork: Regular ring lattice (high clustering, long paths)
- SmallWorldNetwork: Watts-Strogatz model (high clustering, short paths)
- ScaleFreeNetwork: Barabási-Albert model (power-law degree distribution)

This module uses NetworkX for graph generation and analysis, providing
battle-tested implementations and additional network analysis capabilities.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Set, Tuple, Optional
import random
from collections import defaultdict

import networkx as nx


class NetworkTopology(ABC):
    """
    Abstract base class for network topologies.

    Networks determine which agents can interact with each other in the IPD simulation.
    """

    @property
    @abstractmethod
    def num_agents(self) -> int:
        """Number of agents in the network."""
        pass

    @abstractmethod
    def get_neighbors(self, agent_id: int) -> List[int]:
        """
        Get all agents this agent is connected to.

        Args:
            agent_id: ID of the agent

        Returns:
            List of neighbor agent IDs
        """
        pass

    @abstractmethod
    def get_edges(self) -> Iterator[Tuple[int, int]]:
        """
        Iterate over all unique edges in the network.

        Yields:
            Tuples of (agent1_id, agent2_id) for each edge
            Only yields each edge once (i.e., yields (0, 1) but not (1, 0))
        """
        pass

    @abstractmethod
    def has_edge(self, agent1_id: int, agent2_id: int) -> bool:
        """Check if an edge exists between two agents."""
        pass

    def get_degree(self, agent_id: int) -> int:
        """Get the degree (number of connections) of an agent."""
        return len(self.get_neighbors(agent_id))

    def get_degree_distribution(self) -> dict[int, int]:
        """
        Get the degree distribution of the network.

        Returns:
            Dict mapping degree to count of agents with that degree
        """
        distribution: dict[int, int] = defaultdict(int)
        for agent_id in range(self.num_agents):
            degree = self.get_degree(agent_id)
            distribution[degree] += 1
        return dict(distribution)

    def get_clustering_coefficient(self, agent_id: int) -> float:
        """
        Calculate local clustering coefficient for an agent.

        The clustering coefficient is the fraction of possible triangles
        through the agent that exist.

        Args:
            agent_id: ID of the agent

        Returns:
            Clustering coefficient (0.0 to 1.0)
        """
        neighbors = self.get_neighbors(agent_id)
        k = len(neighbors)

        if k < 2:
            return 0.0

        # Count edges between neighbors
        edges_between_neighbors = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1 :]:
                if self.has_edge(n1, n2):
                    edges_between_neighbors += 1

        # Maximum possible edges between neighbors
        max_edges = k * (k - 1) / 2

        return edges_between_neighbors / max_edges

    def get_average_clustering_coefficient(self) -> float:
        """Calculate average clustering coefficient across all agents."""
        total = sum(
            self.get_clustering_coefficient(agent_id)
            for agent_id in range(self.num_agents)
        )
        return total / self.num_agents


class LatticeNetwork(NetworkTopology):
    """
    Regular ring lattice network using NetworkX.

    Each agent is connected to k nearest neighbors (k/2 on each side).
    Creates a ring topology with high clustering and long average path length.

    Example with n=6 agents and k=4:
        Agent 0 connects to: [4, 5, 1, 2]
        Agent 1 connects to: [5, 0, 2, 3]
        etc.
    """

    def __init__(self, num_agents: int, k: int = 4, seed: Optional[int] = None):
        """
        Initialize a ring lattice network.

        Args:
            num_agents: Number of agents in the network
            k: Number of neighbors for each agent (must be even, ≤ num_agents - 1)
            seed: Optional random seed (not used for lattice, but kept for consistency)
        """
        if k % 2 != 0:
            raise ValueError("k must be even")
        if k >= num_agents:
            raise ValueError(f"k ({k}) must be less than num_agents ({num_agents})")

        self._num_agents = num_agents
        self._k = k

        # Use NetworkX Watts-Strogatz with p=0 (no rewiring) to create lattice
        self._graph = nx.watts_strogatz_graph(num_agents, k, p=0, seed=seed)

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def get_neighbors(self, agent_id: int) -> List[int]:
        return sorted(self._graph.neighbors(agent_id))

    def get_edges(self) -> Iterator[Tuple[int, int]]:
        # NetworkX edges are already undirected, return as (min, max) tuples
        for u, v in self._graph.edges():
            yield (min(u, v), max(u, v))

    def has_edge(self, agent1_id: int, agent2_id: int) -> bool:
        return self._graph.has_edge(agent1_id, agent2_id)


class SmallWorldNetwork(NetworkTopology):
    """
    Watts-Strogatz small-world network using NetworkX.

    Starts with a ring lattice and randomly rewires edges with probability p.
    This creates a network with:
    - High clustering (like lattice) when p is low
    - Short average path length (like random graph) when p is moderate

    The "small-world" regime is typically around p = 0.1 to 0.3.

    Reference:
        Watts, D.J. and Strogatz, S.H. (1998).
        "Collective dynamics of 'small-world' networks." Nature, 393(6684), 440-442.
    """

    def __init__(
        self,
        num_agents: int,
        k: int = 4,
        p: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize a Watts-Strogatz small-world network.

        Args:
            num_agents: Number of agents in the network
            k: Number of neighbors for each agent in the initial lattice (must be even)
            p: Probability of rewiring each edge (0 ≤ p ≤ 1)
               p=0: Regular lattice
               p=1: Random graph
               p≈0.1-0.3: Small-world regime
            seed: Optional random seed for reproducibility
        """
        if k % 2 != 0:
            raise ValueError("k must be even")
        if k >= num_agents:
            raise ValueError(f"k ({k}) must be less than num_agents ({num_agents})")
        if not 0 <= p <= 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")

        self._num_agents = num_agents
        self._k = k
        self._p = p

        # Use NetworkX Watts-Strogatz generator
        self._graph = nx.watts_strogatz_graph(num_agents, k, p, seed=seed)

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def get_neighbors(self, agent_id: int) -> List[int]:
        return sorted(self._graph.neighbors(agent_id))

    def get_edges(self) -> Iterator[Tuple[int, int]]:
        # NetworkX edges are already undirected, return as (min, max) tuples
        for u, v in self._graph.edges():
            yield (min(u, v), max(u, v))

    def has_edge(self, agent1_id: int, agent2_id: int) -> bool:
        return self._graph.has_edge(agent1_id, agent2_id)


class ScaleFreeNetwork(NetworkTopology):
    """
    Barabási-Albert scale-free network with custom m0 support.

    Built using preferential attachment: new nodes connect preferentially
    to nodes that already have many connections. This creates a power-law
    degree distribution with "hub" nodes that have many connections.

    Properties:
    - Power-law degree distribution: P(k) ~ k^(-γ) with γ ≈ 3
    - Hub nodes: A few nodes have very high degree
    - Short average path length
    - Low clustering coefficient

    Note: NetworkX's barabasi_albert_graph uses m0=m by default. This implementation
    supports custom m0 values by building the initial complete graph manually.

    Reference:
        Barabási, A.-L. and Albert, R. (1999).
        "Emergence of Scaling in Random Networks." Science, 286(5439), 509-512.
    """

    def __init__(
        self,
        num_agents: int,
        m: int = 2,
        m0: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize a Barabási-Albert scale-free network.

        Args:
            num_agents: Number of agents in the network
            m: Number of edges each new node creates (m ≤ m0)
            m0: Size of initial complete graph
            seed: Optional random seed for reproducibility
        """
        if m > m0:
            raise ValueError(f"m ({m}) must be ≤ m0 ({m0})")
        if m0 > num_agents:
            raise ValueError(f"m0 ({m0}) must be ≤ num_agents ({num_agents})")
        if m < 1:
            raise ValueError("m must be at least 1")

        self._num_agents = num_agents
        self._m = m
        self._m0 = m0
        self._rng = random.Random(seed)

        # Build network with custom m0 support
        self._graph = self._build_scale_free_graph()

    def _build_scale_free_graph(self) -> nx.Graph:
        """
        Build scale-free network with custom m0 using NetworkX graph structure.

        Uses preferential attachment algorithm compatible with NetworkX.
        """
        n = self._num_agents
        graph = nx.Graph()

        # Initialize with a complete graph of m0 nodes
        graph.add_nodes_from(range(self._m0))
        for i in range(self._m0):
            for j in range(i + 1, self._m0):
                graph.add_edge(i, j)

        # Track degree for preferential attachment
        # Each node starts with degree (m0 - 1)
        degrees = [self._m0 - 1] * self._m0
        total_degree = sum(degrees)

        # Add remaining nodes one at a time using preferential attachment
        for new_node in range(self._m0, n):
            graph.add_node(new_node)

            # Select m existing nodes to connect to using preferential attachment
            targets: Set[int] = set()
            attempts = 0
            max_attempts = 100 * self._m  # Prevent infinite loops

            while len(targets) < self._m and attempts < max_attempts:
                attempts += 1

                # Select node with probability proportional to degree
                r = self._rng.random() * total_degree
                cumulative = 0
                for i, degree in enumerate(degrees):
                    cumulative += degree
                    if r <= cumulative:
                        if i not in targets:
                            targets.add(i)
                        break

            # Create edges
            for target in targets:
                graph.add_edge(new_node, target)
                degrees[target] += 1
                total_degree += 2  # Each edge adds 1 to each endpoint

            degrees.append(len(targets))

        return graph

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def get_neighbors(self, agent_id: int) -> List[int]:
        return sorted(self._graph.neighbors(agent_id))

    def get_edges(self) -> Iterator[Tuple[int, int]]:
        # NetworkX edges are already undirected, return as (min, max) tuples
        for u, v in self._graph.edges():
            yield (min(u, v), max(u, v))

    def has_edge(self, agent1_id: int, agent2_id: int) -> bool:
        return self._graph.has_edge(agent1_id, agent2_id)


class WellMixedNetwork(NetworkTopology):
    """
    Well-mixed network where each agent is randomly paired with k others each round.
    The matching changes every round (managed by calling randomize_matching()).
    
    Uses NetworkX's random_regular_graph to generate a new k-regular random graph each round.
    """

    def __init__(self, num_agents: int, k: int = 2, seed: Optional[int] = None):
        if k % 2 != 0:
            raise ValueError("k must be even for random regular graphs")
        if k >= num_agents:
            raise ValueError(f"k ({k}) must be less than num_agents ({num_agents})")
        
        self._num_agents = num_agents
        self._k = k
        self._seed = seed
        self._rng = random.Random(seed)
        self._graph: Optional[nx.Graph] = None
        self.randomize_matching()

    def randomize_matching(self):
        """Generate a new random k-regular graph for this round."""
        # Use a new seed derived from the RNG to ensure different graphs each round
        round_seed = self._rng.randint(0, 2**31)
        self._graph = nx.random_regular_graph(self._k, self._num_agents, seed=round_seed)

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def get_neighbors(self, agent_id: int) -> List[int]:
        return sorted(self._graph.neighbors(agent_id))

    def get_edges(self) -> Iterator[Tuple[int, int]]:
        for u, v in self._graph.edges():
            yield (min(u, v), max(u, v))

    def has_edge(self, agent1_id: int, agent2_id: int) -> bool:
        return self._graph.has_edge(agent1_id, agent2_id)


def create_network(
    network_type: str,
    num_agents: int,
    k: int = 4,
    p: float = 0.1,
    m: int = 2,
    m0: int = 3,
    seed: Optional[int] = None,
) -> NetworkTopology:
    """
    Factory function to create network topologies using NetworkX.

    Args:
        network_type: Type of network ("lattice", "small_world", "scale_free", or "well_mixed")
        num_agents: Number of agents
        k: Neighbors for lattice/small-world/well-mixed
        p: Rewiring probability for small-world
        m: Edges per new node for scale-free
        m0: Initial graph size for scale-free
        seed: Random seed

    Returns:
        NetworkTopology instance
    """
    if network_type == "lattice":
        return LatticeNetwork(num_agents, k=k, seed=seed)
    elif network_type == "small_world":
        return SmallWorldNetwork(num_agents, k=k, p=p, seed=seed)
    elif network_type == "scale_free":
        return ScaleFreeNetwork(num_agents, m=m, m0=m0, seed=seed)
    elif network_type in ("well_mixed", "random_matching"):
        return WellMixedNetwork(num_agents, k=k, seed=seed)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
