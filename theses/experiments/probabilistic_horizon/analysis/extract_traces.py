"""Extract reasoning traces from LiteLLM spend logs for qualitative analysis.

Outputs JSON files with validated reasoning traces for each analysis scenario.
"""

import json
import psycopg2
import sys
from pathlib import Path

DB_CONN = "postgresql://postgres:postgres@localhost:5432/coop_net_litellm"
RESULTS_BASE = Path("theses/experiments/probabilistic_horizon/results/gemini-3-flash-preview_reasoning")
OUTPUT_DIR = RESULTS_BASE / "traces"

# Condition filters: (prompt_filter_sql, time_filter_sql)
# For INFORMED conditions, prompt content uniquely identifies delta.
# For UNINFORMED conditions, we need time windows.
CONDITIONS = {
    "delta0.7_informed": {
        "prompt_filter": "proxy_server_request::text LIKE '%70%% chance the game continues%'",
        "time_filter": None,
        "results_dir": "delta0.7_informed_20260212_205751",
    },
    "delta0.7_uninformed": {
        "prompt_filter": "proxy_server_request::text NOT LIKE '%chance the game continues%'",
        "time_filter": "\"startTime\" >= '2026-02-12 23:57:00' AND \"startTime\" < '2026-02-13 01:05:00'",
        "results_dir": "delta0.7_uninformed_20260212_212942",
    },
    "delta0.8_informed": {
        "prompt_filter": "proxy_server_request::text LIKE '%80%% chance the game continues%'",
        "time_filter": None,
        "results_dir": "delta0.8_informed_20260212_220536",
    },
    "delta0.8_uninformed": {
        "prompt_filter": "proxy_server_request::text NOT LIKE '%chance the game continues%'",
        "time_filter": "\"startTime\" >= '2026-02-13 01:05:00' AND \"startTime\" < '2026-02-13 02:34:00'",
        "results_dir": "delta0.8_uninformed_20260212_224327",
    },
    "delta0.9_informed": {
        "prompt_filter": "proxy_server_request::text LIKE '%90%% chance the game continues%'",
        "time_filter": "\"startTime\" >= '2026-02-12 23:27:00'",  # exclude early test data
        "results_dir": "delta0.9_informed_20260212_233415",
    },
    "delta0.9_uninformed": {
        "prompt_filter": "proxy_server_request::text NOT LIKE '%chance the game continues%'",
        "time_filter": "\"startTime\" >= '2026-02-13 02:34:00'",
        "results_dir": "delta0.9_uninformed_20260213_003122",
    },
}


def get_connection():
    return psycopg2.connect(DB_CONN)


def load_results(condition_name):
    """Load results.json for a condition, return dict keyed by (replication, round, agent_id)."""
    results_dir = CONDITIONS[condition_name]["results_dir"]
    results_path = RESULTS_BASE / results_dir / "results.json"
    with open(results_path) as f:
        data = json.load(f)

    lookup = {}
    for rep in data:
        rep_num = rep["replication"]
        for round_data in rep["history"]:
            rd = round_data["round"]
            for agent_result in round_data["results"]:
                aid = agent_result["agent_id"]
                lookup[(rep_num, rd, aid)] = agent_result["action"]
    return lookup, data


def build_query(condition_name, replication=None, agent_id=None, round_num=None):
    """Build SQL query for extracting traces from a condition."""
    cond = CONDITIONS[condition_name]

    where_clauses = [
        "request_tags::text LIKE '%replication:30%'",
        "model = 'gemini-3-flash-preview'",
        cond["prompt_filter"],
    ]

    if cond["time_filter"]:
        where_clauses.append(cond["time_filter"])

    if replication is not None:
        where_clauses.append(f"request_tags::text LIKE '%replication:{300 + replication}%'")

    if agent_id is not None:
        where_clauses.append(f"request_tags::text LIKE '%agent_id:{agent_id}%'")

    if round_num is not None:
        where_clauses.append(f"request_tags::text LIKE '%round:{round_num}%'")

    where = " AND ".join(where_clauses)

    query = f"""
    SELECT
        SUBSTRING(request_tags::text FROM 'agent_id:(\\d+)')::int as agent_id,
        SUBSTRING(request_tags::text FROM 'round:(\\d+)')::int as round_num,
        SUBSTRING(request_tags::text FROM 'replication:(\\d+)')::int - 300 as replication,
        response->'choices'->0->'message'->>'reasoning_content' as reasoning,
        response->'choices'->0->'message'->>'content' as output,
        "startTime" as timestamp
    FROM "LiteLLM_SpendLogs"
    WHERE {where}
    ORDER BY "startTime" ASC
    """
    return query


def extract_and_validate(conn, condition_name, replication=None, agent_id=None, round_num=None):
    """Extract traces and validate against results.json."""
    results_lookup, _ = load_results(condition_name)
    query = build_query(condition_name, replication, agent_id, round_num)

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()

    traces = []
    validation_errors = 0

    for row in rows:
        aid, rnd, rep, reasoning, output, ts = row

        # Parse action from output — handles both text and JSON formats
        output_clean = output.strip() if output else ""
        parsed_action = None

        # Try JSON format first: {"ACTION": "C"} or {"ACTION": "D"}
        try:
            parsed_json = json.loads(output_clean)
            if isinstance(parsed_json, dict):
                action_val = parsed_json.get("ACTION", parsed_json.get("action", ""))
                if action_val in ("C", "COOPERATE"):
                    parsed_action = "COOPERATE"
                elif action_val in ("D", "DEFECT"):
                    parsed_action = "DEFECT"
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to text parsing
        if parsed_action is None:
            upper = output_clean.upper()
            if "COOPERATE" in upper:
                parsed_action = "COOPERATE"
            elif "DEFECT" in upper:
                parsed_action = "DEFECT"
            else:
                parsed_action = f"UNPARSED: {output_clean[:50]}"

        # Validate against results.json
        expected_action = results_lookup.get((rep, rnd, aid))
        valid = parsed_action == expected_action if expected_action else None

        if valid is False:
            validation_errors += 1
            print(f"  VALIDATION ERROR: {condition_name} rep={rep} round={rnd} agent={aid}: "
                  f"DB={parsed_action}, expected={expected_action}", file=sys.stderr)

        traces.append({
            "agent_id": aid,
            "round": rnd,
            "replication": rep,
            "reasoning": reasoning,
            "output": output.strip() if output else None,
            "parsed_action": parsed_action,
            "expected_action": expected_action,
            "valid": valid,
            "timestamp": str(ts),
        })

    return traces, validation_errors


def extract_deep_case(conn):
    """Scenario A: Full contagion trace for seed=7, delta=0.9 uninformed (15 rounds, 47%→0%)."""
    print("=== Extracting deep case: delta0.9_uninformed, replication=7 ===")
    traces, errors = extract_and_validate(conn, "delta0.9_uninformed", replication=7)
    print(f"  Extracted {len(traces)} traces, {errors} validation errors")
    return traces


def extract_seed_defectors(conn):
    """Scenario B: Round-1 reasoning from each seed defector across all degrading conditions."""
    print("\n=== Extracting seed defector round-1 reasoning ===")

    # Known seed defectors from results.json analysis:
    defectors = {
        "delta0.9_uninformed": {
            7: [20],      # seed=7: agent 20 defects round 1
            6: [6],       # seed=6: agent 6 defects round 1
            9: [12, 15],  # seed=9: agents 12 AND 15 defect round 1
            3: [19],      # seed=3: agent 19 defects round 1 (game ends at round 2)
        },
        "delta0.8_uninformed": {
            1: [],  # need to find
            9: [],  # need to find
        },
        "delta0.7_uninformed": {
            3: [3],   # agent 3 defects round 1
            8: [21],  # agent 21 defects round 1
        },
    }

    # First, find the delta0.8 uninformed defectors
    results_lookup_08, data_08 = load_results("delta0.8_uninformed")
    for rep_data in data_08:
        rep = rep_data["replication"]
        if rep_data["avg_coop_rate"] < 1.0:
            round1 = rep_data["history"][0]
            round1_defectors = [r["agent_id"] for r in round1["results"] if r["action"] == "DEFECT"]
            if round1_defectors:
                defectors["delta0.8_uninformed"][rep] = round1_defectors
                print(f"  Found delta0.8_uninformed rep={rep} round-1 defectors: {round1_defectors}")

    # Also check delta0.9 uninformed for completeness
    results_lookup_09, data_09 = load_results("delta0.9_uninformed")
    for rep_data in data_09:
        rep = rep_data["replication"]
        if rep_data["avg_coop_rate"] < 1.0:
            round1 = rep_data["history"][0]
            round1_defectors = [r["agent_id"] for r in round1["results"] if r["action"] == "DEFECT"]
            print(f"  delta0.9_uninformed rep={rep}: avg={rep_data['avg_coop_rate']:.1f}%, "
                  f"round-1 defectors: {round1_defectors}")

    # Extract traces for all identified seed defectors
    all_defector_traces = {}
    for condition, reps in defectors.items():
        for rep, agents in reps.items():
            if not agents:
                continue
            for agent in agents:
                key = f"{condition}_rep{rep}_agent{agent}"
                print(f"  Extracting {key}...")
                traces, errors = extract_and_validate(conn, condition, replication=rep, agent_id=agent, round_num=1)
                if traces:
                    all_defector_traces[key] = traces[0]
                    print(f"    Output: {traces[0]['output'][:80]}...")
                else:
                    print(f"    WARNING: No trace found!")

    return all_defector_traces, defectors


def extract_informed_contrast(conn):
    """Scenario C: Same agent, same seed, informed vs uninformed contrast."""
    print("\n=== Extracting informed vs uninformed contrast ===")

    # Best contrast: delta0.9, seed=7 (uninformed degrades to 0%, informed stays 100%)
    # Extract agent 20 (the seed defector) from both conditions
    traces = {}

    # Agent 20 from uninformed (defects round 1)
    t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=20, round_num=1)
    traces["uninformed_agent20_r1"] = t[0] if t else None

    # Agent 20 from informed (cooperates round 1)
    t, _ = extract_and_validate(conn, "delta0.9_informed", replication=7, agent_id=20, round_num=1)
    traces["informed_agent20_r1"] = t[0] if t else None

    # Also extract a few more rounds from each for depth
    for rd in [2, 3, 5]:
        t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=20, round_num=rd)
        if t:
            traces[f"uninformed_agent20_r{rd}"] = t[0]
        t, _ = extract_and_validate(conn, "delta0.9_informed", replication=7, agent_id=20, round_num=rd)
        if t:
            traces[f"informed_agent20_r{rd}"] = t[0]

    # Also try a neighbor of agent 20 (agents 19 and 21 on the ring)
    for agent in [19, 21]:
        for rd in [1, 2, 3]:
            t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=agent, round_num=rd)
            if t:
                traces[f"uninformed_agent{agent}_r{rd}"] = t[0]
            t, _ = extract_and_validate(conn, "delta0.9_informed", replication=7, agent_id=agent, round_num=rd)
            if t:
                traces[f"informed_agent{agent}_r{rd}"] = t[0]

    return traces


def extract_cooperating_uninformed(conn):
    """Scenario D: A long uninformed rep maintaining 100% cooperation."""
    print("\n=== Extracting cooperating uninformed rep ===")

    # Find best candidate: long game, 100% cooperation, uninformed
    candidates = []
    for cond in ["delta0.9_uninformed", "delta0.8_uninformed", "delta0.7_uninformed"]:
        _, data = load_results(cond)
        for rep_data in data:
            if rep_data["avg_coop_rate"] == 1.0 and rep_data["actual_rounds"] >= 8:
                candidates.append((cond, rep_data["replication"], rep_data["actual_rounds"]))

    candidates.sort(key=lambda x: -x[2])  # longest first
    print(f"  Candidates (100% coop, 8+ rounds): {candidates}")

    if not candidates:
        print("  No suitable candidate found!")
        return {}

    best = candidates[0]
    cond, rep, rounds = best
    print(f"  Selected: {cond} rep={rep} ({rounds} rounds)")

    # Extract a sample of agents across rounds (not all — too much data)
    traces = {}

    # Round 1: sample 3 agents
    for agent in [0, 12, 24]:
        t, _ = extract_and_validate(conn, cond, replication=rep, agent_id=agent, round_num=1)
        if t:
            traces[f"agent{agent}_r1"] = t[0]

    # Middle and late rounds: sample agent 0
    mid = rounds // 2
    for rd in [mid, rounds - 1, rounds]:
        t, _ = extract_and_validate(conn, cond, replication=rep, agent_id=0, round_num=rd)
        if t:
            traces[f"agent0_r{rd}"] = t[0]

    # Also get a different agent in late rounds
    t, _ = extract_and_validate(conn, cond, replication=rep, agent_id=12, round_num=rounds)
    if t:
        traces[f"agent12_r{rounds}"] = t[0]

    traces["_metadata"] = {"condition": cond, "replication": rep, "total_rounds": rounds}
    return traces


def extract_two_seed_case(conn):
    """Scenario E: delta0.9 uninformed seed=9 — agents 12 AND 15 both defect round 1."""
    print("\n=== Extracting two-seed case: delta0.9_uninformed, replication=9 ===")

    traces = {}

    # Both seed defectors round 1
    for agent in [12, 15]:
        t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=9, agent_id=agent, round_num=1)
        if t:
            traces[f"agent{agent}_r1"] = t[0]

    # Their immediate neighbors who get infected
    # On ring lattice k=2: agent 12's neighbors are 11 and 13, agent 15's neighbors are 14 and 16
    # Between them: agents 13 and 14 are caught between the two wavefronts
    for rd in [1, 2, 3]:
        for agent in [11, 13, 14, 16]:
            t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=9, agent_id=agent, round_num=rd)
            if t:
                traces[f"agent{agent}_r{rd}"] = t[0]

    # Extract full contagion trajectory
    full_traces, errors = extract_and_validate(conn, "delta0.9_uninformed", replication=9)
    traces["_full_trace_count"] = len(full_traces)
    traces["_validation_errors"] = errors

    # Build round-by-round defector map
    defector_map = {}
    for tr in full_traces:
        rd = tr["round"]
        if tr["parsed_action"] == "DEFECT":
            defector_map.setdefault(rd, []).append(tr["agent_id"])
    traces["_defector_map"] = {str(k): sorted(v) for k, v in sorted(defector_map.items())}

    return traces


def extract_deep_case_key_moments(conn):
    """Extract key moments from the deep case for the narrative."""
    print("\n=== Extracting deep case key moments ===")

    _, data = load_results("delta0.9_uninformed")
    rep7_data = next(r for r in data if r["replication"] == 7)

    # Build round-by-round defector map from results.json
    defector_map = {}
    for round_data in rep7_data["history"]:
        rd = round_data["round"]
        defectors = [r["agent_id"] for r in round_data["results"] if r["action"] == "DEFECT"]
        defector_map[rd] = defectors

    print(f"  Defector map: { {k: v for k, v in sorted(defector_map.items())} }")

    # Key agents/moments to extract:
    # Round 1: agent 20 (seed defector)
    # Round 2: agents 19, 21 (first wave)
    # Round 3: agents 18, 22 (second wave)
    # ...propagation continues...
    # Last round: identify last cooperators switching

    # Find the wavefront progression
    all_defectors = set()
    wavefront = {}
    for rd in sorted(defector_map.keys()):
        new_defectors = set(defector_map[rd]) - all_defectors
        if new_defectors:
            wavefront[rd] = sorted(new_defectors)
        all_defectors.update(defector_map[rd])

    print(f"  Wavefront: { {k: v for k, v in sorted(wavefront.items())} }")

    # Extract reasoning for key wavefront moments
    traces = {"_defector_map": {str(k): v for k, v in sorted(defector_map.items())},
              "_wavefront": {str(k): v for k, v in sorted(wavefront.items())}}

    # Seed defector: agent 20, round 1
    t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=20, round_num=1)
    if t:
        traces["seed_defector_r1"] = t[0]

    # First-wave infection: agents 19, 21 in rounds 1-3
    for agent in [19, 21]:
        for rd in range(1, 4):
            t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=agent, round_num=rd)
            if t:
                traces[f"wave1_agent{agent}_r{rd}"] = t[0]

    # Second-wave: agents 18, 22 — their switch round
    for agent in [18, 22]:
        for rd in range(1, 5):
            t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=agent, round_num=rd)
            if t:
                traces[f"wave2_agent{agent}_r{rd}"] = t[0]

    # Mid-game frontier
    for rd in [5, 7, 10]:
        for aid in sorted(wavefront.get(rd, [])):
            t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=aid, round_num=rd)
            if t:
                traces[f"frontier_agent{aid}_r{rd}"] = t[0]
            # Also get the round before they switched
            t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=aid, round_num=rd-1)
            if t:
                traces[f"frontier_agent{aid}_r{rd-1}"] = t[0]

    # Last holdouts — find who cooperated longest
    last_round = max(defector_map.keys())
    last_round_defectors = set(defector_map.get(last_round, []))
    prev_round_defectors = set(defector_map.get(last_round - 1, []))
    last_holdouts = last_round_defectors - prev_round_defectors

    if last_holdouts:
        for agent in sorted(last_holdouts):
            for rd in [last_round - 2, last_round - 1, last_round]:
                t, _ = extract_and_validate(conn, "delta0.9_uninformed", replication=7, agent_id=agent, round_num=rd)
                if t:
                    traces[f"last_holdout_agent{agent}_r{rd}"] = t[0]

    # Also find who was still cooperating in second-to-last round
    if last_round - 1 in defector_map:
        still_coop = set(range(25)) - prev_round_defectors
        print(f"  Last holdouts (switched in round {last_round}): {sorted(last_holdouts)}")
        print(f"  Still cooperating at round {last_round-1}: {sorted(still_coop)}")

    return traces


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()

    try:
        # Scenario A: Deep case
        deep_traces = extract_deep_case(conn)
        with open(OUTPUT_DIR / "deep_case_full.json", "w") as f:
            json.dump(deep_traces, f, indent=2, default=str)

        key_moments = extract_deep_case_key_moments(conn)
        with open(OUTPUT_DIR / "deep_case_key_moments.json", "w") as f:
            json.dump(key_moments, f, indent=2, default=str)

        # Scenario B: Seed defectors
        defector_traces, defector_map = extract_seed_defectors(conn)
        with open(OUTPUT_DIR / "seed_defectors.json", "w") as f:
            json.dump({"traces": defector_traces, "defector_map": {k: {str(rk): rv for rk, rv in v.items()} for k, v in defector_map.items()}}, f, indent=2, default=str)

        # Scenario C: Informed vs uninformed contrast
        contrast_traces = extract_informed_contrast(conn)
        with open(OUTPUT_DIR / "informed_contrast.json", "w") as f:
            json.dump(contrast_traces, f, indent=2, default=str)

        # Scenario D: Cooperating uninformed
        coop_traces = extract_cooperating_uninformed(conn)
        with open(OUTPUT_DIR / "cooperating_uninformed.json", "w") as f:
            json.dump(coop_traces, f, indent=2, default=str)

        # Scenario E: Two-seed case
        two_seed_traces = extract_two_seed_case(conn)
        with open(OUTPUT_DIR / "two_seed_case.json", "w") as f:
            json.dump(two_seed_traces, f, indent=2, default=str)

        print("\n=== All traces extracted successfully ===")
        print(f"Output directory: {OUTPUT_DIR}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
