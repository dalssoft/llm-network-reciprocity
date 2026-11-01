"""
Experiment Set B: Parameter Sweep (Setting 2).

Goal: Replicate Han et al. (2024) Figures 4 and 5.
- Test whether LLMs recognize when b/c > k should favor cooperation.
- Setting 2: 25 agents, 15 rounds, k ∈ {2,4,6}, b/c ∈ {2,4,6}

From paper: "Fewer rounds, More participants"

Features:
- Parallel execution of replications.
- Progress bars using tqdm.
- CLI arguments for model selection and cache control.
- Comprehensive error logging for debugging failed runs.
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

from .config import HanConfig, Action, PayoffMatrix, SETTING_2
from .agents.agent import LLMAgent
from .environments.pd_env import NetworkedPD
from ..common.network import create_network
from src.infra.llm.client import LLMClient
from src.infra.llm.models import Models

# Setup experiment logger
experiment_logger = logging.getLogger("experiment_b")
experiment_logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# File handler for detailed experiment logs
_log_filename = f"logs/experiment_b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_file_handler = logging.FileHandler(_log_filename)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s'
))
experiment_logger.addHandler(_file_handler)

# Console handler for important messages
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
experiment_logger.addHandler(_console_handler)


def run_replication(
    rep_idx: int,
    k: int,
    bc_ratio: float,
    num_agents: int,
    num_rounds: int,
    pm: PayoffMatrix,
    model: Models,
    disable_cache: bool,
    pbar_position: int,
    with_reasoning: bool = False,
) -> Dict[str, Any]:
    """Run a single replication of a parameter condition."""

    experiment_logger.info(
        f"Starting replication {rep_idx+1} | model={model.model_name} | "
        f"k={k} | bc={bc_ratio} | reasoning={with_reasoning}"
    )

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
            seed=rep_idx + 200,
        )

        # 3. Create agents
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
            )
            for i in range(num_agents)
        ]

        # 4. Create network (always lattice for Setting 2)
        experiment_logger.debug(f"Rep {rep_idx+1}: Creating lattice network with k={k}")
        network = create_network(
            network_type="lattice", num_agents=num_agents, k=k, seed=config.seed
        )

        # 5. Initialize environment with callback
        env = NetworkedPD(agents, network, config, progress_callback=progress_callback)

        # 6. Run simulation
        experiment_logger.debug(f"Rep {rep_idx+1}: Starting simulation ({num_rounds} rounds)")
        history = env.run_simulation()

        # 7. Compute summary stats
        coop_rates = []
        for round_data in history:
            actions = [r["action"] for r in round_data["results"]]
            coop_rate = sum(1 for a in actions if a == Action.COOPERATE) / len(actions)
            coop_rates.append(coop_rate)

        experiment_logger.info(
            f"Rep {rep_idx+1} COMPLETED | final_coop={coop_rates[-1]:.1%} | "
            f"avg_coop={sum(coop_rates)/len(coop_rates):.1%}"
        )

        return {
            "replication": rep_idx,
            "history": history,
            "coop_rates": coop_rates,
            "final_coop_rate": coop_rates[-1],
            "avg_coop_rate": sum(coop_rates) / len(coop_rates),
        }
    except Exception as e:
        # Log detailed error information
        error_msg = (
            f"Rep {rep_idx+1} FAILED | model={model.model_name} | "
            f"k={k} | bc={bc_ratio} | reasoning={with_reasoning}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {str(e)}\n"
            f"Stack trace:\n{traceback.format_exc()}"
        )
        experiment_logger.error(error_msg)
        raise  # Re-raise to be caught by the executor
    finally:
        pbar.close()


def run_condition(
    k: int,
    bc_ratio: float,
    num_agents: int,
    num_rounds: int,
    replications: int,
    model: Models,
    disable_cache: bool,
    max_workers: int,
    with_reasoning: bool = False,
) -> str:
    """Run a single parameter condition with parallel replications."""
    condition_name = f"bc{bc_ratio}_k{k}"

    # Theoretical prediction
    theory_prediction = (
        "cooperation favored"
        if bc_ratio > k
        else ("boundary" if bc_ratio == k else "defection favored")
    )

    print(
        f"\n{'='*60}\n"
        f"Running Condition: {condition_name}\n"
        f"  Network: lattice, Agents: {num_agents}, k={k}\n"
        f"  Rounds: {num_rounds}, b/c={bc_ratio}, Replications: {replications}\n"
        f"  Theory: b/c {'>' if bc_ratio > k else ('=' if bc_ratio == k else '<')} k → {theory_prediction}\n"
        f"{'='*60}"
    )

    pm = PayoffMatrix.from_bc_ratio(bc_ratio)
    print(f"  Payoff Matrix: T={pm.T}, R={pm.R}, P={pm.P}, S={pm.S}")
    print(f"  Cost (c): {pm.T - pm.R}, Benefit (b): {pm.T - pm.P}")
    print(f"  Parallel workers: {max_workers}")

    results = []
    failed_replications = []
    error_details = []  # Store detailed error info

    experiment_logger.info(
        f"Starting condition: {condition_name} | model={model.model_name} | "
        f"replications={replications} | workers={max_workers}"
    )

    # Use ThreadPoolExecutor for parallelism (LLM calls are I/O bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_replication,
                rep_idx=i,
                k=k,
                bc_ratio=bc_ratio,
                num_agents=num_agents,
                num_rounds=num_rounds,
                pm=pm,
                model=model,
                disable_cache=disable_cache,
                pbar_position=i + 1,
                with_reasoning=with_reasoning,
            ): i
            for i in range(replications)
        }

        # Main tqdm bar to track overall completion
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
                        f"\n    Rep {rep_idx+1} finished. Final coop: {rep_result['final_coop_rate']:.1%}, Avg: {rep_result['avg_coop_rate']:.1%}"
                    )
                except Exception as e:
                    failed_replications.append(rep_idx)
                    # Capture detailed error information
                    error_info = {
                        "replication": rep_idx,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    error_details.append(error_info)
                    
                    pbar.update(1)
                    print(f"\n    Rep {rep_idx+1} FAILED: {type(e).__name__}: {e}")
                    print(f"    (Continuing with remaining replications...)")
                    
                    # Log to experiment logger (stack trace already logged in run_replication)
                    experiment_logger.warning(
                        f"Replication {rep_idx+1} failed for {model.model_name}/{condition_name}: "
                        f"{type(e).__name__}: {e}"
                    )

    # Save results (even if some replications failed)
    reasoning_suffix = "_reasoning" if with_reasoning else ""
    output_dir = f"theses/experiments/modern_models/results/experiment_b/{model.model_name}{reasoning_suffix}/{condition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Report completion status
    completed_count = len(results)
    failed_count = len(failed_replications)
    print(f"\n  Completed replications: {completed_count}/{replications}")
    if failed_count > 0:
        print(f"  Failed replications: {failed_count} (indices: {failed_replications})")
        print(f"  ⚠️  PARTIAL RESULTS - Saving {completed_count} successful replications")

    with open(f"{output_dir}/results.json", "w") as f:

        def default_encoder(obj):
            if isinstance(obj, Action):
                return obj.name
            raise TypeError

        json.dump(results, f, default=default_encoder, indent=2)

    # Save config with error details
    config_sample = HanConfig(model=model)
    config_data = {
        "experiment": "Experiment B: Parameter Sweep (Setting 2)",
        "paper_reference": "Han et al. (2024), Figures 4 & 5",
        "timestamp": datetime.now().isoformat(),
        "condition": condition_name,
        "network_type": "lattice",
        "num_agents": num_agents,
        "k": k,
        "num_rounds": num_rounds,
        "bc_ratio": bc_ratio,
        "replications": replications,
        "completed_replications": completed_count,
        "failed_replications": failed_count,
        "failed_replication_indices": failed_replications,
        "model": model.model_name,
        "model_provider": model.provider,
        "temperature": config_sample.temperature,
        "cache_disabled": disable_cache,
        "with_reasoning": with_reasoning,
        "payoff_matrix": {
            "T": pm.T,
            "R": pm.R,
            "P": pm.P,
            "S": pm.S,
            "b": pm.T - pm.P,
            "c": pm.T - pm.R,
        },
        "theory_prediction": theory_prediction,
        "tutorial_enabled": True,
        "feedback_format": "Han et al. (2024)",
    }
    
    # Add error details if there were failures
    if error_details:
        config_data["errors"] = error_details
        experiment_logger.warning(
            f"Condition {condition_name} completed with {failed_count} failures. "
            f"Error details saved to config.json"
        )
    
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n  Results saved to {output_dir}")
    if failed_count > 0:
        print(f"  ⚠️  Note: Only {completed_count}/{replications} replications completed successfully")
        print(f"  You can re-run this condition to get more data if needed")
        print(f"  📋 Error details saved in config.json")
    
    experiment_logger.info(
        f"Condition {condition_name} finished | "
        f"completed={completed_count}/{replications} | "
        f"output={output_dir}"
    )
    
    return output_dir


def main():
    """
    Run Setting 2: Parameter sweep over k and b/c.

    Han's Setting 2 parameters:
    - 25 agents
    - 15 rounds
    - k ∈ {2, 4, 6}
    - b/c ∈ {2, 4, 6}
    - 5 replications each
    """
    parser = argparse.ArgumentParser(
        description="Run Experiment B (Han Setting 2 - Parameter Sweep)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to use (default: gpt-3.5-turbo). Options: gpt-3.5-turbo, gpt-4-0613, gpt-4o-mini, etc.",
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
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Subset of k values to run (default: all from SETTING_2)",
    )
    parser.add_argument(
        "--bc",
        type=float,
        nargs="+",
        default=None,
        help="Subset of b/c values to run (default: all from SETTING_2)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable reasoning/extended thinking mode for supported models",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Number of replications per condition (default: from SETTING_2, which is 5)",
    )
    args = parser.parse_args()

    load_dotenv()

    # Cache is disabled by default
    disable_cache = not args.enable_cache

    # Parse model
    try:
        model = Models.from_model_name(args.model)
    except ValueError:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {[m.model_name for m in Models]}")
        return

    NUM_AGENTS = SETTING_2["num_agents"]
    NUM_ROUNDS = SETTING_2["num_rounds"]
    KS = args.k if args.k else SETTING_2["ks"]
    BC_RATIOS = args.bc if args.bc else SETTING_2["bc_ratios"]
    REPLICATIONS = args.replications if args.replications else SETTING_2["replications"]

    total_conditions = len(KS) * len(BC_RATIOS)

    print(f"\n{'#'*60}")
    print("# Modern Models Experiment - Setting 2")
    print("# Parameter Sweep")
    print(f"{'#'*60}")
    print(f"Model: {model.model_name}")
    print(f"Reasoning: {'enabled' if args.reasoning else 'disabled'}")
    print(f"Cache: {'enabled' if args.enable_cache else 'disabled'}")
    print(f"Parallelism: {args.max_workers} workers")
    print(f"Conditions: k ∈ {KS}, b/c ∈ {BC_RATIOS} ({total_conditions} total)")
    print(
        f"Each condition: {NUM_AGENTS} agents × {NUM_ROUNDS} rounds × {REPLICATIONS} reps"
    )

    # Estimate API calls
    total_decisions = total_conditions * REPLICATIONS * NUM_AGENTS * NUM_ROUNDS
    print(f"Total agent decisions: {total_decisions:,}")
    print(f"📋 Log file: {_log_filename}")
    
    experiment_logger.info(
        f"{'='*60}\n"
        f"EXPERIMENT START\n"
        f"Model: {model.model_name}\n"
        f"Reasoning: {args.reasoning}\n"
        f"Cache disabled: {disable_cache}\n"
        f"Workers: {args.max_workers}\n"
        f"Conditions: {total_conditions} (k ∈ {KS}, b/c ∈ {BC_RATIOS})\n"
        f"Total decisions: {total_decisions:,}\n"
        f"{'='*60}"
    )

    completed = 0
    for k in KS:
        for bc in BC_RATIOS:
            completed += 1
            print(f"\n[Condition {completed}/{total_conditions}]")
            run_condition(
                k=k,
                bc_ratio=bc,
                num_agents=NUM_AGENTS,
                num_rounds=NUM_ROUNDS,
                replications=REPLICATIONS,
                model=model,
                disable_cache=disable_cache,
                max_workers=args.max_workers,
                with_reasoning=args.reasoning,
            )

    print(f"\n{'#'*60}")
    print("# Experiment B Complete!")
    print(f"# Ran {total_conditions} conditions with {REPLICATIONS} replications each")
    print(f"{'#'*60}")
    print(f"📋 Full logs available at: {_log_filename}")
    
    experiment_logger.info(
        f"{'='*60}\n"
        f"EXPERIMENT COMPLETE\n"
        f"Model: {model.model_name}\n"
        f"Conditions completed: {total_conditions}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
