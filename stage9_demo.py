"""
stage9_demo.py
--------------
Stage 9 Demonstration Script

Demonstrates generating CandidateExits from existing FloorPlans, and 
constructing ExitConfigurations from those candidates. 

This script does NOT run evacuation simulations. It proves the decoupling
and combination generation scaling.
"""

import json
import sys
from pathlib import Path

# Ensure RL_ENV is in sys.path
_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.generation import (
    generate_candidate_exits,
    generate_exit_configurations,
)


def run_demo() -> None:
    print("=" * 80)
    print(" STAGE 9: CANDIDATE EXIT GENERATION DEMO")
    print("=" * 80)
    print()

    scenarios_dir = Path("scenarios")
    scenario_files = ["basic_building.json", "bottleneck.json"]

    for filename in scenario_files:
        filepath = scenarios_dir / filename
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue
            
        with filepath.open("r") as f:
            data = json.load(f)
            
        # Parse the JSON into our models
        config = SimulationConfig(**data)
        floor_plan = config.floor_plan
        
        print(f"Floor plan: {config.scenario_name} ({filepath.name})")
        print(f"  Grid size: {floor_plan.grid.rows}x{floor_plan.grid.cols}")
        print(f"  Walls: {len(floor_plan.walls)}")
        
        # 1. Generate Candidates
        candidates = generate_candidate_exits(floor_plan)
        print(f"  Candidate exits (boundary, non-wall): {len(candidates)}")
        
        if not candidates:
            print("  (No candidates available)")
            print("-" * 80)
            continue
            
        # 2. Generate Configurations
        # Let's see how many configurations for N=1 and N=2 exits
        for requested_exits in [1, 2]:
            if requested_exits > len(candidates):
                continue
                
            configs = list(generate_exit_configurations(candidates, requested_exits))
            print(f"  Requested exits: {requested_exits}")
            print(f"  Generated configurations: {len(configs)}")
            
            # Print the first 3 just to show what they look like
            for i, cfg in enumerate(configs[:3]):
                print(f"    - {cfg.configuration_id}: {cfg.exits}")
            if len(configs) > 3:
                print(f"    - ... and {len(configs) - 3} more")
                
        print("-" * 80)
        
        
if __name__ == "__main__":
    run_demo()
