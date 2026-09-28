"""
update_visualizer_data.py
-------------------------
Simulates scenarios (basic_building.json and bottleneck.json) and exports
step-by-step state traces, including Stage 10 explicit door metadata and widths,
to simulation_data.json and directly into visualize.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure RL_ENV is in sys.path
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.generation import generate_candidate_exits
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy


def record_scenario_simulation(config_path: Path) -> dict:
    config = SimulationConfig.from_json(config_path)
    strategy = ShortestPathStrategy()
    sim = Simulation(config, movement_strategy=strategy)

    candidate_exits = generate_candidate_exits(config.floor_plan)

    scenario_record = {
        "scenario_name": config.scenario_name,
        "grid": {"rows": config.floor_plan.grid.rows, "cols": config.floor_plan.grid.cols},
        "walls": [list(w) for w in config.floor_plan.walls],
        "exits": [list(e) for e in config.exit_configuration.exits],
        "exit_widths": [config.exit_configuration.get_width(e) for e in config.exit_configuration.exits],
        "doors": [
            {
                "position": list(d.position),
                "width": d.width,
                "exterior": d.exterior,
                "door_id": d.door_id,
            }
            for d in config.floor_plan.doors
        ],
        "candidate_exits": [
            {
                "position": list(c.position),
                "width": c.width,
                "door_id": c.door_id,
            }
            for c in candidate_exits
        ],
        "capacity": config.parameters.default_cell_capacity,
        "max_timesteps": config.parameters.max_timesteps,
        "initial_agents": [
            {"agent_id": a.agent_id, "row": a.row, "col": a.col}
            for a in config.occupants.agents
        ],
        "steps": [],
    }

    # Step 0 (initial state)
    initial_step_agents = []
    for a in sim.agents:
        p = strategy._find_nearest_exit_path(a.position, sim.environment, sim.nav_graph)
        path = [list(pt) for pt in p] if p else None
        initial_step_agents.append({
            "agent_id": a.agent_id,
            "row": a.position[0],
            "col": a.position[1],
            "state": a.state.name,
            "waiting_steps": 0,
            "path": path,
        })

    occ_cells = sim.occupancy.get_occupied_cells()
    scenario_record["steps"].append({
        "timestep": 0,
        "agents": initial_step_agents,
        "occupancy": [[f"{r},{c}", count] for (r, c), count in occ_cells.items()],
        "waiting_ids": [],
        "trace": [],
    })

    # Run loop step by step
    while not sim.state.is_terminated:
        should_continue = sim.step()
        trace = sim.last_step_trace
        step_agents = []
        waiting_ids = []

        for a in sim.agents:
            path = None
            if a.is_active:
                p = strategy._find_nearest_exit_path(a.position, sim.environment, sim.nav_graph)
                if p:
                    path = [list(pt) for pt in p]
            w_steps = sim.metrics._waiting_steps.get(a.agent_id, 0)
            if w_steps > 0:
                waiting_ids.append(a.agent_id)

            step_agents.append({
                "agent_id": a.agent_id,
                "row": a.position[0],
                "col": a.position[1],
                "state": a.state.name,
                "evacuation_timestep": a.evacuation_timestep,
                "waiting_steps": w_steps,
                "path": path,
            })

        step_trace_items = []
        if trace:
            for r in trace.agent_results:
                step_trace_items.append({
                    "agent_id": r.agent_id,
                    "from": list(r.position_before),
                    "to": list(r.position_after),
                    "req": list(r.requested) if r.requested else None,
                    "result": r.result,
                    "reason": r.rejection_reason.name if r.rejection_reason else None,
                })

        post_occ = sim.occupancy.get_occupied_cells()
        scenario_record["steps"].append({
            "timestep": sim.state.timestep,
            "agents": step_agents,
            "occupancy": [[f"{r},{c}", count] for (r, c), count in post_occ.items()],
            "waiting_ids": waiting_ids,
            "trace": step_trace_items,
        })

        if not should_continue:
            break

    return scenario_record


def main() -> None:
    scenarios = ["bottleneck.json", "basic_building.json"]
    sim_data = {}

    for name in scenarios:
        p = Path("scenarios") / name
        if p.exists():
            key = p.stem
            print(f"Generating simulation trace for {key}...")
            sim_data[key] = record_scenario_simulation(p)

    out_file = Path("simulation_data.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(sim_data, f, indent=2)
    print(f"Saved {out_file} ({out_file.stat().st_size} bytes)")

    # Also update embedded SIM_DATA in visualize.html
    html_path = Path("visualize.html")
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        marker = "const SIM_DATA = "
        idx = content.find(marker)
        if idx != -1:
            end_idx = content.find(";\n", idx)
            if end_idx != -1:
                json_str = json.dumps(sim_data)
                new_content = content[:idx + len(marker)] + json_str + content[end_idx:]
                html_path.write_text(new_content, encoding="utf-8")
                print(f"Updated embedded SIM_DATA in {html_path}")


if __name__ == "__main__":
    main()
