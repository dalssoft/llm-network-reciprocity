"""
Probabilistic Horizon Experiment.

Tests whether a known endpoint is necessary for Gemini 3 Flash (R)'s
endgame collapse by replacing fixed rounds with stochastic continuation.

Design:
- Model: Gemini 3 Flash (R) by default
- Setting: 25 agents, k=2, lattice network
- Delta values: continuation probability per round after minimum
- Two info conditions: informed (told about delta) vs uninformed (no duration info)

Usage:
    # Default: all deltas, both conditions, 10 reps
    python -m theses.experiments.probabilistic_horizon.run_experiment

    # Subset
    python -m theses.experiments.probabilistic_horizon.run_experiment --delta 0.9 --horizon-info informed --replications 3

    # Different model
    python -m theses.experiments.probabilistic_horizon.run_experiment --model gemini-3-flash-preview --no-reasoning
"""

import os
import json
import argparse
import concurrent.futures
import logging
import traceback
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

from .config import ProbHorizonConfig, Action, PayoffMatrix, HorizonInfo
from .agents.agent import LLMAgent
from .environments.pd_env import NetworkedPD
from ..common.network import create_network
from src.infra.llm.client import LLMClient
from src.infra.llm.models import Models

# Setup experiment logger
experiment_logger = logging.getLogger("prob_horizon")
experiment_logger.setLevel(logging.DEBUG)

os.makedirs("logs", exist_ok=True)

_log_filename = f"logs/prob_horizon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_file_handler = logging.FileHandler(_log_filename)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s'
))
experiment_logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
experiment_logger.addHandler(_console_handler)


def run_replication(
    rep_idx: int,
    delta: float,
    horizon_info: HorizonInfo,
    num_agents: int,
    k: int,
    bc_ratio: float,
    pm: PayoffMatrix,
    model: Models,
    disable_cache: bool,
    pbar_position: int,
    with_reasoning: bool = True,
    min_rounds: int = 1,
    max_rounds: int = 100,
) -> Dict[str, Any]:
    """Run a single replication of a condition."""

    condition_label = f"d={delta}_{horizon_info.value}"
    experiment_logger.info(
        f"Starting replication {rep_idx+1} | {condition_label} | "
        f"model={model.model_name} | reasoning={with_reasoning}"
    )

    # Progress bar — use max_rounds as upper bound (actual may be shorter)
    pbar = tqdm(
        total=max_rounds,
        desc=f"  Rep {rep_idx+1:2d}",
        position=pbar_position,
        leave=False,
    )

    def progress_callback(current_round: int, total_rounds: int):
        pbar.update(1)

    try:
        config = ProbHorizonConfig(
            num_agents=num_agents,
            k=k,
            payoff_matrix=pm,
            model=model,
            continuation_delta=delta,
            min_rounds=min_rounds,
            max_rounds=max_rounds,
            horizon_info=horizon_info,
            with_reasoning=with_reasoning,
            seed=rep_idx + 300,  # Offset from other experiments
        )

        # Create agents
        experiment_logger.debug(f"Rep {rep_idx+1}: Creating LLM client and {num_agents} agents")
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
                with_reasoning=with_reasoning,
                horizon_info=horizon_info,
                continuation_delta=delta,
            )
            for i in range(num_agents)
        ]

        # Create network (always lattice)
        experiment_logger.debug(f"Rep {rep_idx+1}: Creating lattice network with k={k}")
        network = create_network(
            network_type="lattice", num_agents=num_agents, k=k, seed=config.seed
        )

        # Run simulation
        env = NetworkedPD(agents, network, config, progress_callback=progress_callback)
        experiment_logger.debug(f"Rep {rep_idx+1}: Starting simulation (delta={delta}, max={max_rounds})")
        history = env.run_simulation()

        actual_rounds = len(history)

        # Compute summary stats
        coop_rates = []
        for round_data in history:
            actions = [r["action"] for r in round_data["results"]]
            coop_rate = sum(1 for a in actions if a == Action.COOPERATE) / len(actions)
            coop_rates.append(coop_rate)

        experiment_logger.info(
            f"Rep {rep_idx+1} COMPLETED | {condition_label} | "
            f"actual_rounds={actual_rounds} | final_coop={coop_rates[-1]:.1%} | "
            f"avg_coop={sum(coop_rates)/len(coop_rates):.1%}"
        )

        return {
            "replication": rep_idx,
            "history": history,
            "coop_rates": coop_rates,
            "final_coop_rate": coop_rates[-1],
            "avg_coop_rate": sum(coop_rates) / len(coop_rates),
            "actual_rounds": actual_rounds,
        }
    except Exception as e:
        error_msg = (
            f"Rep {rep_idx+1} FAILED | {condition_label} | "
            f"model={model.model_name}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {str(e)}\n"
            f"Stack trace:\n{traceback.format_exc()}"
        )
        experiment_logger.error(error_msg)
        raise
    finally:
        pbar.close()


def run_condition(
    delta: float,
    horizon_info: HorizonInfo,
    num_agents: int,
    k: int,
    bc_ratio: float,
    replications: int,
    model: Models,
    disable_cache: bool,
    max_workers: int,
    with_reasoning: bool = True,
    min_rounds: int = 1,
    max_rounds: int = 100,
) -> str:
    """Run a single condition (delta x horizon_info) with parallel replications."""
    condition_name = f"delta{delta}_{horizon_info.value}"

    expected_rounds = min_rounds + 1 / (1 - delta) if delta < 1.0 else max_rounds
    pm = PayoffMatrix.from_bc_ratio(bc_ratio)

    print(
        f"\n{'='*60}\n"
        f"Running Condition: {condition_name}\n"
        f"  Network: lattice, Agents: {num_agents}, k={k}, b/c={bc_ratio}\n"
        f"  Delta: {delta}, Horizon info: {horizon_info.value}\n"
        f"  Expected rounds: ~{expected_rounds:.1f} (min={min_rounds}, max={max_rounds})\n"
        f"  Replications: {replications}\n"
        f"{'='*60}"
    )

    print(f"  Payoff Matrix: T={pm.T}, R={pm.R}, P={pm.P}, S={pm.S}")
    print(f"  Parallel workers: {max_workers}")

    results = []
    failed_replications = []
    error_details = []

    experiment_logger.info(
        f"Starting condition: {condition_name} | model={model.model_name} | "
        f"replications={replications} | workers={max_workers}"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_replication,
                rep_idx=i,
                delta=delta,
                horizon_info=horizon_info,
                num_agents=num_agents,
                k=k,
                bc_ratio=bc_ratio,
                pm=pm,
                model=model,
                disable_cache=disable_cache,
                pbar_position=i + 1,
                with_reasoning=with_reasoning,
                min_rounds=min_rounds,
                max_rounds=max_rounds,
            ): i
            for i in range(replications)
        }

        with tqdm(
            total=replications, desc=f"Overall {condition_name}", position=0
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                rep_idx = futures[future]
                try:
                    rep_result = future.result()
                    results.append(rep_result)
                    pbar.update(1)

                    print(
                        f"\n    Rep {rep_idx+1} finished. "
                        f"Rounds: {rep_result['actual_rounds']}, "
                        f"Final coop: {rep_result['final_coop_rate']:.1%}, "
                        f"Avg: {rep_result['avg_coop_rate']:.1%}"
                    )
                except Exception as e:
                    failed_replications.append(rep_idx)
                    error_details.append({
                        "replication": rep_idx,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": datetime.now().isoformat(),
                    })
                    pbar.update(1)
                    print(f"\n    Rep {rep_idx+1} FAILED: {type(e).__name__}: {e}")

    # Save results
    reasoning_suffix = "_reasoning" if with_reasoning else ""
    output_dir = (
        f"theses/experiments/probabilistic_horizon/results/"
        f"{model.model_name}{reasoning_suffix}/"
        f"{condition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    completed_count = len(results)
    failed_count = len(failed_replications)
    print(f"\n  Completed replications: {completed_count}/{replications}")
    if failed_count > 0:
        print(f"  Failed replications: {failed_count} (indices: {failed_replications})")

    with open(f"{output_dir}/results.json", "w") as f:
        def default_encoder(obj):
            if isinstance(obj, Action):
                return obj.name
            raise TypeError
        json.dump(results, f, default=default_encoder, indent=2)

    # Compute actual rounds stats
    actual_rounds_list = [r["actual_rounds"] for r in results]
    rounds_stats = {
        "mean": sum(actual_rounds_list) / len(actual_rounds_list) if actual_rounds_list else 0,
        "min": min(actual_rounds_list) if actual_rounds_list else 0,
        "max": max(actual_rounds_list) if actual_rounds_list else 0,
        "all": actual_rounds_list,
    }

    config_data = {
        "experiment": "Probabilistic Horizon Experiment",
        "timestamp": datetime.now().isoformat(),
        "condition": condition_name,
        "network_type": "lattice",
        "num_agents": num_agents,
        "k": k,
        "bc_ratio": bc_ratio,
        "continuation_delta": delta,
        "horizon_info": horizon_info.value,
        "min_rounds": min_rounds,
        "max_rounds": max_rounds,
        "expected_rounds": expected_rounds,
        "actual_rounds_stats": rounds_stats,
        "replications": replications,
        "completed_replications": completed_count,
        "failed_replications": failed_count,
        "failed_replication_indices": failed_replications,
        "model": model.model_name,
        "model_provider": model.provider,
        "temperature": 1.0,
        "cache_disabled": disable_cache,
        "with_reasoning": with_reasoning,
        "payoff_matrix": {
            "T": pm.T, "R": pm.R, "P": pm.P, "S": pm.S,
            "b": pm.T - pm.P, "c": pm.T - pm.R,
        },
    }
    if error_details:
        config_data["errors"] = error_details

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n  Results saved to {output_dir}")
    experiment_logger.info(
        f"Condition {condition_name} finished | "
        f"completed={completed_count}/{replications} | "
        f"rounds_mean={rounds_stats['mean']:.1f} | output={output_dir}"
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run Probabilistic Horizon Experiment"
    )
    parser.add_argument(
        "--model", type=str, default="gemini-3-flash-preview",
        help="Model to use (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--no-reasoning", action="store_true",
        help="Disable reasoning mode (default: reasoning enabled)",
    )
    parser.add_argument(
        "--delta", type=float, nargs="+", default=[0.7, 0.8, 0.9],
        help="Continuation probability values (default: 0.7 0.8 0.9)",
    )
    parser.add_argument(
        "--horizon-info", type=str, nargs="+",
        default=["informed", "uninformed"],
        choices=["informed", "uninformed"],
        help="Horizon info conditions (default: both)",
    )
    parser.add_argument(
        "--bc", type=float, default=4.0,
        help="b/c ratio (default: 4.0)",
    )
    parser.add_argument(
        "--k", type=int, default=2,
        help="Network degree (default: 2)",
    )
    parser.add_argument(
        "--num-agents", type=int, default=25,
        help="Number of agents (default: 25)",
    )
    parser.add_argument(
        "--replications", type=int, default=10,
        help="Replications per condition (default: 10)",
    )
    parser.add_argument(
        "--min-rounds", type=int, default=1,
        help="Minimum rounds before stochastic termination (default: 1)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=100,
        help="Safety cap on number of rounds (default: 100)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help="Number of parallel replications (default: 5)",
    )
    parser.add_argument(
        "--enable-cache", action="store_true",
        help="Enable LiteLLM cache (default: disabled)",
    )
    args = parser.parse_args()

    load_dotenv()

    disable_cache = not args.enable_cache
    with_reasoning = not args.no_reasoning

    try:
        model = Models.from_model_name(args.model)
    except ValueError:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {[m.model_name for m in Models]}")
        return

    deltas = args.delta
    horizon_infos = [HorizonInfo(h) for h in args.horizon_info]
    total_conditions = len(deltas) * len(horizon_infos)

    print(f"\n{'#'*60}")
    print("# Probabilistic Horizon Experiment")
    print(f"{'#'*60}")
    print(f"Model: {model.model_name}")
    print(f"Reasoning: {'enabled' if with_reasoning else 'disabled'}")
    print(f"Cache: {'enabled' if args.enable_cache else 'disabled'}")
    print(f"Parallelism: {args.max_workers} workers")
    print(f"Deltas: {deltas}")
    print(f"Horizon info: {[h.value for h in horizon_infos]}")
    print(f"Conditions: {total_conditions} total")
    print(f"b/c={args.bc}, k={args.k}, {args.num_agents} agents")
    print(f"Min rounds: {args.min_rounds}, Max rounds: {args.max_rounds}")
    print(f"Replications per condition: {args.replications}")
    print(f"Log file: {_log_filename}")

    experiment_logger.info(
        f"{'='*60}\n"
        f"EXPERIMENT START\n"
        f"Model: {model.model_name}, Reasoning: {with_reasoning}\n"
        f"Deltas: {deltas}, Horizon info: {[h.value for h in horizon_infos]}\n"
        f"Conditions: {total_conditions}, Replications: {args.replications}\n"
        f"{'='*60}"
    )

    completed = 0
    for delta in deltas:
        for hi in horizon_infos:
            completed += 1
            print(f"\n[Condition {completed}/{total_conditions}]")
            run_condition(
                delta=delta,
                horizon_info=hi,
                num_agents=args.num_agents,
                k=args.k,
                bc_ratio=args.bc,
                replications=args.replications,
                model=model,
                disable_cache=disable_cache,
                max_workers=args.max_workers,
                with_reasoning=with_reasoning,
                min_rounds=args.min_rounds,
                max_rounds=args.max_rounds,
            )

    print(f"\n{'#'*60}")
    print("# Probabilistic Horizon Experiment Complete!")
    print(f"# Ran {total_conditions} conditions with {args.replications} replications each")
    print(f"{'#'*60}")
    print(f"Full logs: {_log_filename}")

    experiment_logger.info(
        f"{'='*60}\n"
        f"EXPERIMENT COMPLETE\n"
        f"Conditions completed: {total_conditions}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
