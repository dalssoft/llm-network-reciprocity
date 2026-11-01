# llm-network-reciprocity

Code and data for the master's dissertation:

> Lojudice Sobrinho, D. (2026). *Reciprocidade em rede e agentes LLM: dinâmica
> de cooperação no dilema do prisioneiro iterado* [Network reciprocity and LLM
> agents: cooperation dynamics in the iterated prisoner's dilemma]. Master's
> dissertation, Escola de Artes, Ciências e Humanidades, Universidade de São
> Paulo. Advisor: Prof. Dr. Camilo Rodrigues Neto.

LLM agents play the iterated prisoner's dilemma on a network. Each agent is a
language model that sees its neighbors' last moves and chooses to cooperate
(C) or defect (D). The study asks whether LLM agents use network structure to
sustain cooperation, as network reciprocity theory predicts (cooperation
should grow when b/c > k) and as humans do in lab experiments.

The repository holds the simulation code, the prompts, the analysis scripts,
and the raw data of all 536 replications reported in the dissertation.

## What is here

| Stage | Folder | Models | Replications | Dissertation |
|---|---|---|---|---|
| 1. Reproduction of Han et al. | `theses/experiments/han_reproduction/` | GPT-3.5, GPT-4 | 65 (Exp A: 20, Exp B: 45) | Sec. 5.1 |
| 2. Recent models | `theses/experiments/modern_models/` | Gemini 3 Flash, Gemini 2.5 Flash Lite, GPT-5-mini, GPT-OSS-120B, DeepSeek-V3.2 (with and without reasoning where available) | 411 (Exp A: 80, Exp B: 331) | Sec. 5.2 to 5.5 |
| 3. Probabilistic horizon | `theses/experiments/probabilistic_horizon/` | Gemini 3 Flash with reasoning | 60 | Sec. 5.6 |
| Behavioral model | `theses/experiments/says_vs_does/` | all of the above | uses all 536 | Sec. 5.7 |

The two experiment designs follow Han et al. (2025), which in turn follow the
human study of Rand et al. (2014):

- **Experiment A**: 8 agents, 25 rounds, a fixed ring (each agent has the same
  2 neighbors every round) versus a well-mixed population (partners reshuffled
  every round).
- **Experiment B**: 25 agents, 15 rounds, a fixed ring lattice with k ∈ {2, 4, 6}
  neighbors and benefit-to-cost ratio b/c ∈ {2, 4, 6}.
- **Probabilistic horizon**: like Experiment B (k = 2, b/c = 4), but after each
  round the game goes on with probability δ ∈ {0.7, 0.8, 0.9}. Agents are
  either told this rule (informed) or not (uninformed). 10 replications per
  condition.

Shared code lives in `theses/experiments/common/network.py` (the network
topologies) and `src/infra/` (the LLM client).

Each stage folder has the same layout:

```
agents/agent.py          LLM agent: conversation history, tutorial, round feedback
environments/pd_env.py   game loop: neighbors, decisions, payoffs
prompts/templates.py     the exact prompts (the Han reproduction keeps Han's
                         wording, typos included)
config.py                game parameters and payoff matrix
run_*.py                 entry points that run simulations (need an LLM API)
analysis/                scripts that read the saved results (no API needed)
results/                 raw data, one folder per condition run
```

## Data format

Each run folder in `results/` has two files:

- `config.json`: the run parameters (model, reasoning on or off, network
  type, k, b/c, number of agents and rounds, temperature, payoff matrix, and
  how many replications completed or failed).
- `results.json`: a list with one entry per replication. Each entry has the
  full round-by-round history (every agent's action, payoff and neighbors),
  the cooperation rate per round, and summary rates.

Some conditions of Experiment B have more than one run folder, because runs
were split across several sessions (API quotas). The analysis scripts merge
them.

`probabilistic_horizon/results/.../traces/` holds the reasoning traces quoted
in Chapter 5 (Gemini 3 Flash). They were extracted from the LiteLLM proxy logs
with `analysis/extract_traces.py`; the other traces are not part of this
repository. That script reads the proxy's PostgreSQL log database, which is not
public, and needs `psycopg2`, which is not in `requirements.txt`. It is kept
to show how the files were made.

`says_vs_does/data/decision_table.csv.gz` flattens all 536 replications into
one row per decision (171,050 rows). It is rebuilt from the `results/`
folders by `build_decision_table.py`.

## Reproduce the analyses (no API key needed)

Python 3.11 or newer. From the repository root:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Stage 1: GPT-3.5 Experiment B heatmap (Fig. 7) and threshold table
python -m theses.experiments.han_reproduction.analysis.analyze_experiment_b
# Experiment A summary; picks the latest runs (GPT-4) unless you pass the
# GPT-3.5 folders with --well-mixed and --structured
python -m theses.experiments.han_reproduction.analysis.analyze_results

# Stage 2: recent models, Experiment B (Fig. 10) and Experiment A summaries
python -m theses.experiments.modern_models.analysis.analyze_experiment_b_comprehensive
python -m theses.experiments.modern_models.analysis.analyze_results --all

# Stage 3: probabilistic horizon (Figs. 13 and 14) and the schematic (Fig. 5)
python -m theses.experiments.probabilistic_horizon.analysis.analyze_results
python -m theses.experiments.probabilistic_horizon.analysis.analyze_contagion
python -m theses.experiments.probabilistic_horizon.analysis.plot_contagion_schematic

# Sec. 5.7: decision table, behavioral model, period-2 test, Fig. 15
python -m theses.experiments.says_vs_does.build_decision_table
python -m theses.experiments.says_vs_does.fit_behavioral_model
python -m theses.experiments.says_vs_does.measure_period2
python -m theses.experiments.says_vs_does.make_figures
```

Figures are written next to the data they come from (in the `results/`
folders, or `says_vs_does/figures/`).

### Figure map

| Fig. | File in the dissertation | Made by |
|---|---|---|
| 5 | `fig_contagio_esquema.png` | `probabilistic_horizon/analysis/plot_contagion_schematic.py` (a hand-built example, not data) |
| 6 | `fig_exp_a_gpt35.png` | `han_reproduction/analysis/plots.py`, function `plot_experiment_a`, with the two `gpt-3.5-turbo` runs of `results/experiment_a/` |
| 7 | `fig_heatmap_gpt35.png` | `han_reproduction/analysis/analyze_experiment_b.py` → `cooperation_heatmap.png` |
| 10 | `fig_heatmaps_modernos.png` | `modern_models/analysis/analyze_experiment_b_comprehensive.py` → `heatmaps.png` |
| 13 | `fig_prob_horizon_trajectories.png` | `probabilistic_horizon/analysis/analyze_contagion.py` → `cooperation_trajectories.png` |
| 14 | `fig_contagion_spacetime.png` | `probabilistic_horizon/analysis/analyze_contagion.py` → `contagion_spacetime.png` |
| 15 | `fig_reactive_rules.png` | `says_vs_does/make_figures.py` |

Figures 8, 9, 11 and 12 were made with one-off code that is not included; the
data behind them is here. Figures 1 to 4 are diagrams, not data.

Small visual differences from the printed figures are expected: some plots
draw bootstrap confidence bands, which change a little on every run.

## Run new simulations (needs API keys)

The code calls models through an OpenAI-compatible endpoint. We used a local
[LiteLLM](https://github.com/BerriAI/litellm) proxy, which routes each model
name to its provider. All runs used the Docker image
`ghcr.io/berriai/litellm:v1.80.10.rc.4`:

```bash
cp .env.example .env        # fill in the provider keys
docker run --rm -p 4000:4000 --env-file .env \
  -v "$PWD/litellm_config.yaml:/app/config.yaml" \
  ghcr.io/berriai/litellm:v1.80.10.rc.4 --config /app/config.yaml --port 4000
```

Then, in another terminal (each command accepts `--help`):

```bash
# Stage 1
python -m theses.experiments.han_reproduction.run_experiment_a --model gpt-3.5-turbo
python -m theses.experiments.han_reproduction.run_experiment_b

# Stage 2
python -m theses.experiments.modern_models.run_all_models          # Experiment A, all models
python -m theses.experiments.modern_models.run_experiment_b --model gemini-3-flash-preview
python -m theses.experiments.modern_models.run_experiment_b_batch  # Experiment B, the 3 models used

# Stage 3
python -m theses.experiments.probabilistic_horizon.run_experiment
```

New runs are saved as new folders in `results/`. Model outputs are not
deterministic, and providers update or retire models (GPT-3.5 and GPT-4-0613
may no longer be served), so new runs will not match the saved data exactly.
All runs reported in the dissertation had the LiteLLM response cache off
(`"cache_disabled": true` in every `config.json`). The proxy config in this
repository has no cache, so the runners' `--enable-cache` option has no effect
with it.

## Not included

- An early batch of Experiment A with recent models (42 replications) that was
  replaced by the runs in `modern_models/results/experiment_a/`. The
  dissertation does not use it.
- The original data of Han et al. (2025); see their paper.

## License

Code: MIT (see `LICENSE`). Data (everything under `results/` and `data/`
folders): CC BY 4.0 (see `LICENSE-DATA`).

## Citation

See `CITATION.cff`, or cite the dissertation above.

## References

- Han, J., Battu, B., Romić, I., Rahwan, T., & Holme, P. (2025). Static network
  structure cannot stabilize cooperation among large language model agents.
  *PLOS ONE*, 20(5), e0320094. https://doi.org/10.1371/journal.pone.0320094
- Rand, D. G., Nowak, M. A., Fowler, J. H., & Christakis, N. A. (2014). Static
  network structure can stabilize human cooperation. *PNAS*, 111(48),
  17093–17098. https://doi.org/10.1073/pnas.1400406111
