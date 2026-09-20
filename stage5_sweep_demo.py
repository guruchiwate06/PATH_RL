"""Behavioral verification script for Stage 5 — run from evacuation_simulation/."""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

_PARENT = Path(__file__).parent.parent.resolve()
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.experiment import (
    SweepPoint, aggregate_results, export_csv, export_json, run_scenario, run_sweep
)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
OUTPUT_DIR = Path(__file__).parent / "experiment_output"
OUTPUT_DIR.mkdir(exist_ok=True)
SEEDS = [10, 20, 30, 40, 50]

basic_config = SimulationConfig.from_json(SCENARIOS_DIR / "basic_building.json")
bottleneck_config = SimulationConfig.from_json(SCENARIOS_DIR / "bottleneck.json")

sweep_points = [
    SweepPoint("basic_6", basic_config,
               {"scenario": "basic_building", "agent_count": len(basic_config.agents)}),
    SweepPoint("bottleneck_12", bottleneck_config,
               {"scenario": "bottleneck", "agent_count": len(bottleneck_config.agents)}),
]

t0 = time.perf_counter()
experiment = run_sweep(sweep_points, SEEDS, experiment_id="stage5_sweep_demo")
elapsed = time.perf_counter() - t0

expected_runs = 2 * len(SEEDS)
print(f"[RUN COUNT]  {experiment.run_count} runs  (expected {expected_runs})  PASS={experiment.run_count == expected_runs}")

print("\nPer-run results:")
for rec in experiment.runs:
    r = rec.result
    sc = rec.params["scenario"]
    met = f"{r.mean_evacuation_time:.1f}" if r.mean_evacuation_time is not None else "None"
    print(f"  {sc:<22}  seed={rec.seed:<3}  evac={r.evacuated_count}/{r.total_agents}  mean_t={met:<6}  wait={r.total_waiting_steps}")

basic_records = [r for r in experiment.runs if r.params["scenario"] == "basic_building"]
bott_records  = [r for r in experiment.runs if r.params["scenario"] == "bottleneck"]
ba = aggregate_results(basic_records)
bn = aggregate_results(bott_records)

print(f"\n[BASIC AGG]  mean_evac_t={ba.mean_evacuation_time_mean:.2f}  std={ba.mean_evacuation_time_std:.4f}  wait_mean={ba.total_waiting_steps_mean:.2f}  cong_steps={ba.congested_cell_steps_mean:.2f}")
print(f"[BOTT  AGG]  mean_evac_t={bn.mean_evacuation_time_mean:.2f}  std={bn.mean_evacuation_time_std:.4f}  wait_mean={bn.total_waiting_steps_mean:.2f}  cong_steps={bn.congested_cell_steps_mean:.2f}")

# Reproducibility
ra = run_scenario(bottleneck_config, seed=42)
rb = run_scenario(bottleneck_config, seed=42)
same = (ra.result.total_timesteps == rb.result.total_timesteps
        and ra.result.evacuated_count == rb.result.evacuated_count
        and ra.result.mean_evacuation_time == rb.result.mean_evacuation_time)
print(f"\n[REPRO]  seed=42 x2: identical={same}  timesteps={ra.result.total_timesteps}/{rb.result.total_timesteps}  mean_t={ra.result.mean_evacuation_time}/{rb.result.mean_evacuation_time}")

# Config isolation
print(f"[ISOLATION]  basic agents={len(basic_config.agents)} (expect 6)  bottleneck agents={len(bottleneck_config.agents)} (expect 12)  PASS={len(basic_config.agents)==6 and len(bottleneck_config.agents)==12}")

# Aggregate manual verification
bott_met = [r.result.mean_evacuation_time for r in bott_records]
non_null = [v for v in bott_met if v is not None]
manual_mean = sum(non_null) / len(non_null)
agg_match = abs(bn.mean_evacuation_time_mean - manual_mean) < 1e-9
print(f"[AGG VERIFY] values={bott_met}  agg={bn.mean_evacuation_time_mean:.4f}  manual={manual_mean:.4f}  match={agg_match}")

# Export
csv_p = export_csv(experiment.runs, OUTPUT_DIR / "sweep_results.csv")
json_p = export_json(experiment, OUTPUT_DIR / "sweep_experiment.json")

with csv_p.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
csv_ok = (len(rows) == expected_runs
          and "agent_count" in rows[0]
          and "mean_evacuation_time" in rows[0])
print(f"\n[CSV]  rows={len(rows)} (expect {expected_runs})  has_agent_count={'agent_count' in rows[0]}  has_mean_evac={'mean_evacuation_time' in rows[0]}  PASS={csv_ok}")

with json_p.open(encoding="utf-8") as f:
    jd = json.load(f)
json_ok = (jd["run_count"] == expected_runs
           and jd["aggregate"] is not None
           and "mean_evacuation_time_mean" in jd["aggregate"])
print(f"[JSON]  run_count={jd['run_count']} (expect {expected_runs})  agg_present={jd['aggregate'] is not None}  agg_mean_t={jd['aggregate']['mean_evacuation_time_mean']}  PASS={json_ok}")

print(f"\n[PERF]  {experiment.run_count} runs  total={elapsed:.3f}s  avg={elapsed/experiment.run_count:.4f}s/run")
print(f"CSV  -> {csv_p}")
print(f"JSON -> {json_p}")
print("\nALL VERIFICATIONS COMPLETE")
