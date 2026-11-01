"""
Experiment Set A: Well-Mixed vs. Structured Networks (Setting 1).

Goal: Replicate Han et al. (2024) Figure 3.
- Verify that LLMs show opposite cooperation patterns compared to humans.
- Setting 1: 8 agents, 25 rounds, k=2, b/c=6

Features:
- Parallel execution of replications.
- Progress bars using tqdm.
- Progress callbacks from environment to UI layer.
"""

import os
import json
import argparse
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv

from .config import HanConfig, Action, PayoffMatrix, SETTING_1
from .agents.agent import LLMAgent
from .environments.pd_env import NetworkedPD
from ..common.network import create_network
from src.infra.llm.client import LLMClient
from src.infra.llm.models import Models


def run_replication(
    rep_idx: int,
    network_type: str,
    num_agents: int,
    k: int,
    num_rounds: int,
    bc_ratio: float,
    pm: PayoffMatrix,
    model: Models,
    disable_cache: bool,
    pbar_position: int,
) -> Dict[str, Any]:
    """Run a single replication of an experimental condition."""

    # 1. Progress bar setup
    pbar = tqdm(
        total=num_rounds,
        desc=f"  Rep {rep_idx+1:2d}",
        position=pbar_position,
        leave=False,
    )

    def progress_callback(current_round: int, total_rounds: int):
        pbar.update(1)

    try:
        # 2. Setup config
        config = HanConfig(
            num_agents=num_agents,
            k=k,
            num_rounds=num_rounds,
            payoff_matrix=pm,
            model=model,
            seed=rep_idx + 100,
        )

        # 3. Create agents
        llm_client = LLMClient()
        agents = [
            LLMAgent(
                agent_id=i,
                name=f"Agent_{i}",
                llm_client=llm_client,
                model=model,
                temperature=config.temperature,
                payoff_matrix=pm,
                disable_cache=disable_cache,
            )
            for i in range(num_agents)
        ]

        # 4. Create network
        network = create_network(
            network_type=network_type, num_agents=num_agents, k=k, seed=config.seed
        )

        # 5. Initialize environment with callback
        env = NetworkedPD(agents, network, config, progress_callback=progress_callback)

        # 6. Run simulation
        history = env.run_simulation()

        # 7. Compute summary stats
        coop_rates = []
        for round_data in history:
            actions = [r["action"] for r in round_data["results"]]
            coop_rate = sum(1 for a in actions if a == Action.COOPERATE) / len(actions)
            coop_rates.append(coop_rate)

        return {
            "replication": rep_idx,
            "history": history,
            "coop_rates": coop_rates,
            "final_coop_rate": coop_rates[-1],
            "avg_coop_rate": sum(coop_rates) / len(coop_rates),
        }
    finally:
        pbar.close()


def run_condition(
    condition_name: str,
    network_type: str,
    num_agents: int,
    k: int,
    num_rounds: int,
    replications: int,
    model: Models,
    bc_ratio: float = 6.0,
    disable_cache: bool = False,
    max_workers: int = 5,
):
    """Run a single experimental condition with parallel replications."""
    print(
        f"\n{'='*60}\n"
        f"Running Condition: {condition_name}\n"
        f"  Network: {network_type}, Agents: {num_agents}, k={k}\n"
        f"  Rounds: {num_rounds}, b/c={bc_ratio}, Replications: {replications}\n"
        f"{'='*60}"
    )

    pm = PayoffMatrix.from_bc_ratio(bc_ratio)
    print(f"  Payoff Matrix: T={pm.T}, R={pm.R}, P={pm.P}, S={pm.S}")
    print(f"  Cost (c): {pm.T - pm.R}, Benefit (b): {pm.T - pm.P}")
    print(f"  Parallel workers: {max_workers}")

    results = []

    # We use a ThreadPoolExecutor for parallelism since LLM calls are I/O bound
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each replication
        # We use position=i+1 to stack bars below the main status
        futures = {
            executor.submit(
                run_replication,
                rep_idx=i,
                network_type=network_type,
                num_agents=num_agents,
                k=k,
                num_rounds=num_rounds,
                bc_ratio=bc_ratio,
                pm=pm,
                model=model,
                disable_cache=disable_cache,
                pbar_position=i + 1,
            ): i
            for i in range(replications)
        }

        # Use a main tqdm bar to track overall completion
        with tqdm(
            total=replications, desc=f"Overall {condition_name}", position=0
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    rep_result = future.result()
                    results.append(rep_result)
                    pbar.update(1)

                    rep_idx = futures[future]
                    print(
                        f"\n    Rep {rep_idx+1} finished. Final coop: {rep_result['final_coop_rate']:.1%}, Avg: {rep_result['avg_coop_rate']:.1%}"
                    )
                except Exception as e:
                    print(f"\n    Rep {futures[future]+1} failed: {e}")

    # 8. Save results
    output_dir = f"theses/experiments/han_reproduction/results/experiment_a/{condition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/results.json", "w") as f:

        def default_encoder(obj):
            if isinstance(obj, Action):
                return obj.name
            raise TypeError

        json.dump(results, f, default=default_encoder, indent=2)

    # Save config
    config_sample = HanConfig(model=model)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(
            {
                "experiment": "Experiment A: Well-Mixed vs Structured (Setting 1)",
                "paper_reference": "Han et al. (2024), Figure 3",
                "timestamp": datetime.now().isoformat(),
                "condition": condition_name,
                "network_type": network_type,
                "num_agents": num_agents,
                "k": k,
                "num_rounds": num_rounds,
                "bc_ratio": bc_ratio,
                "replications": replications,
                "model": model.model_name,
                "model_provider": model.provider,
                "temperature": config_sample.temperature,
                "cache_disabled": disable_cache,
                "payoff_matrix": {
                    "T": pm.T,
                    "R": pm.R,
                    "P": pm.P,
                    "S": pm.S,
                    "b": pm.T - pm.P,
                    "c": pm.T - pm.R,
                },
                "tutorial_enabled": True,
                "feedback_format": "Han et al. (2024)",
            },
            f,
            indent=2,
        )

    print(f"\n  Results saved to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run Experiment A (Han Setting 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to use (default: gpt-3.5-turbo). Options: gpt-3.5-turbo, gpt-4-0613, gpt-4o, etc.",
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable LiteLLM cache (default: disabled for reproducibility)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel replications to run (default: 5)",
    )
    args = parser.parse_args()

    # Cache is disabled by default
    disable_cache = not args.enable_cache

    load_dotenv()

    # Parse model
    from src.infra.llm.models import Models

    try:
        model = Models.from_model_name(args.model)
    except ValueError:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {[m.model_name for m in Models]}")
        return

    NUM_AGENTS = SETTING_1["num_agents"]
    NUM_ROUNDS = SETTING_1["num_rounds"]
    K = SETTING_1["k"]
    BC_RATIO = SETTING_1["bc_ratio"]
    REPLICATIONS = SETTING_1["replications"]

    print(f"\n{'#'*60}")
    print("# Han et al. (2024) Reproduction - Setting 1")
    print("# Well-Mixed vs. Structured Networks (Figure 3)")
    print(f"{'#'*60}")
    print(f"Model: {model.model_name}")
    print(f"Cache: {'enabled' if args.enable_cache else 'disabled'}")
    print(f"Parallelism: {args.max_workers} workers")

    # Condition 1: Well-Mixed
    run_condition(
        condition_name="well_mixed",
        network_type="well_mixed",
        num_agents=NUM_AGENTS,
        k=K,
        num_rounds=NUM_ROUNDS,
        replications=REPLICATIONS,
        model=model,
        bc_ratio=BC_RATIO,
        disable_cache=disable_cache,
        max_workers=args.max_workers,
    )

    # Condition 2: Structured (Ring Network)
    run_condition(
        condition_name="structured_ring",
        network_type="lattice",
        num_agents=NUM_AGENTS,
        k=K,
        num_rounds=NUM_ROUNDS,
        replications=REPLICATIONS,
        model=model,
        bc_ratio=BC_RATIO,
        disable_cache=disable_cache,
        max_workers=args.max_workers,
    )

    print(f"\n{'#'*60}")
    print("# Experiment A Complete!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
