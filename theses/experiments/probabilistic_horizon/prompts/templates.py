"""
Prompt templates for probabilistic horizon experiment.

Based on Han et al. (2024) prompts, modified to replace fixed round count
with stochastic continuation information (or no duration info at all).
"""

from typing import List, Dict, Any, Optional
from ..config import Action, PayoffMatrix, HorizonInfo


def build_tutorial_messages(
    payoff_matrix: PayoffMatrix,
    is_well_mixed: bool = False,
    horizon_info: HorizonInfo = HorizonInfo.INFORMED,
    continuation_delta: float = 0.9,
) -> List[Dict[str, str]]:
    """
    Build the tutorial conversation that precedes the game.

    Modified from Han's original: the sentence about fixed round count is
    replaced based on horizon_info condition:
    - INFORMED: told about continuation probability, no fixed endpoint
    - UNINFORMED: no duration information at all
    """
    cost_per_neighbor = payoff_matrix.T - payoff_matrix.R  # c = 10
    benefit_per_neighbor = payoff_matrix.T - payoff_matrix.P  # b = 60 for b/c=6

    neighbor_text = (
        "Your neighbors will be randomized between every round."
        if is_well_mixed
        else "You will be connected to the same people every round."
    )

    # Build the duration sentence based on condition
    if horizon_info == HorizonInfo.INFORMED:
        pct = continuation_delta * 100
        duration_text = (
            f"After each round, there is a {pct:.0f}% chance the game continues "
            f"to the next round. There is no predetermined number of rounds."
        )
    else:
        # UNINFORMED: no duration information
        duration_text = ""

    # Build the first user message with or without duration info
    first_user_parts = [
        "The game will be played over a series of rounds. In every round you make a choice "
        "about whether to pay to give points to the other players you are connected to."
    ]
    if duration_text:
        first_user_parts.append(duration_text)
    first_user_parts.append(neighbor_text)
    first_user_parts.append("We now describe the game in more details. Are you ready?")

    first_user_content = " ".join(first_user_parts)

    messages = [
        {
            "role": "system",
            "content": "Pretend you are a human playing a prisoner's dilemma game in a lab environment. I will now explain the rules to you, are you ready?",
        },
        {
            "role": "user",
            "content": first_user_content,
        },
        {
            "role": "user",
            "content": (
                f"In every round you choose whether to pay to give points to the people you are connected to. "
                f"If you choose to cooperate (represented by 'C'), you pay {cost_per_neighbor} points for each player "
                f"you are connected to and each of them gain {benefit_per_neighbor} points. "
                f"If you choose to defect (represented by 'D'), you do not pay any points and do not change the "
                f"points of the players you are connected to. Each player you are connected to has the same choice. "
                f"For each of them that chooses 'C', you gain {benefit_per_neighbor} points. "
                f"Once everyone makes a decision, I will tell you the result of the round. "
                f"You will be shown the choices of each player you are connected to and how many points in total "
                f"you gained or lost. You will also be shown how many points each player you are connected to "
                f"gained or lost in total. These numbers are affected by your choice, their choice, and also the "
                f"choices of any other players connected by them who may or may not be connected to you. "
                f"Remember, for every 300 points you have at the end of the game, we will pay you 1 dollar. "
                f"You should aim to get as many points as you can. Do you understand?"
            ),
        },
        {
            "role": "user",
            "content": "You have now completed the tutorial. Are you ready?",
        },
        {
            "role": "user",
            "content": "Thank you for completing the tutorial. You will now be playing with other participants. Please only give your choice with 'D' or 'C'. Are you ready?",
        },
    ]
    return messages


def build_round_prompt(
    round_num: int,
    last_round_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the prompt for a specific round.

    Han's format shows only LAST round info, not full history.
    Unchanged from original — no round count mentioned here.
    """
    if round_num == 1:
        return "Please make choice for the first round. Please only reply your choice with 'D' or 'C'. "

    if last_round_info is None:
        return "Please choose for the current round. "

    # Build last round summary
    my_action = last_round_info["my_action"]
    my_paid = last_round_info["my_paid"]
    my_gained = last_round_info["my_gained"]
    my_total = last_round_info["my_total"]

    prompt_parts = [
        f"Last round you chose {my_action}, paid {my_paid} points, and gained {my_gained} points, "
        f"for a total of {my_total} points."
    ]

    # Add neighbor info
    for neighbor_info in last_round_info.get("neighbors", []):
        n_id = neighbor_info["id"]
        n_action = neighbor_info["action"]
        n_paid = neighbor_info["paid"]
        n_gained_from_me = neighbor_info["gained_from_me"]
        n_gained_from_others = neighbor_info["gained_from_others"]
        n_total = neighbor_info["total"]

        prompt_parts.append(
            f"Last round, neighbor {n_id} chose {n_action}, paid {n_paid} points, "
            f"and gained {n_gained_from_me} points from you, and {n_gained_from_others} points from other players, "
            f"for a total of {n_total} points."
        )

    prompt_parts.append("Please choose for the current round. ")
    return "".join(prompt_parts)


def compute_last_round_info(
    agent_id: int,
    my_action: Action,
    neighbor_actions: Dict[int, Action],
    payoff_matrix: PayoffMatrix,
    all_agent_actions: Dict[int, Action],  # All agents' actions this round
    network_neighbors: Dict[int, List[int]],  # agent_id -> list of neighbor_ids
) -> Dict[str, Any]:
    """
    Compute the detailed last round info needed for Han's prompt format.
    """
    cost = payoff_matrix.T - payoff_matrix.R  # c
    benefit = payoff_matrix.T - payoff_matrix.P  # b

    neighbor_ids = list(neighbor_actions.keys())
    k = len(neighbor_ids)

    # My payoff calculation
    my_paid = cost * k if my_action == Action.COOPERATE else 0
    my_gained = sum(
        benefit
        for n_id, n_action in neighbor_actions.items()
        if n_action == Action.COOPERATE
    )
    my_total = my_gained - my_paid

    # Neighbor info
    neighbors_info = []
    for n_id in neighbor_ids:
        n_action = neighbor_actions[n_id]
        n_neighbors = network_neighbors.get(n_id, [])
        n_k = len(n_neighbors)

        # Neighbor paid cost if they cooperated
        n_paid = cost * n_k if n_action == Action.COOPERATE else 0

        # What neighbor gained FROM ME (benefit if I cooperated, 0 otherwise)
        n_gained_from_me = benefit if my_action == Action.COOPERATE else 0

        # What neighbor gained from OTHER neighbors (not me)
        n_other_neighbors = [x for x in n_neighbors if x != agent_id]
        n_gained_from_others = sum(
            benefit
            for other_id in n_other_neighbors
            if all_agent_actions.get(other_id) == Action.COOPERATE
        )

        n_total = n_gained_from_me + n_gained_from_others - n_paid

        neighbors_info.append(
            {
                "id": n_id,
                "action": "C" if n_action == Action.COOPERATE else "D",
                "paid": n_paid,
                "gained_from_me": n_gained_from_me,
                "gained_from_others": n_gained_from_others,
                "total": n_total,
            }
        )

    return {
        "my_action": "C" if my_action == Action.COOPERATE else "D",
        "my_paid": my_paid,
        "my_gained": my_gained,
        "my_total": my_total,
        "neighbors": neighbors_info,
    }
