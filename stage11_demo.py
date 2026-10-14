"""
stage11_demo.py
---------------
Stage 11 Demonstration: Regulatory & Exit-Placement Constraint Validation.

Demonstrates:
1. Candidate exterior doors discovered from FloorPlan.
2. Interior passages NOT treated as exit candidates.
3. Valid exit configuration checked — PASS result.
4. Invalid configuration checked — structured violations produced.
5. Action mask API demonstrated.
6. All results are machine-readable (Pydantic models).

DISCLAIMER: Research/design-assistance tool only. Not legal certification.
Results mean "compliant with encoded constraints in the selected regulation
profile", not "legally compliant".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from evacuation_simulation.simulation.config import SimulationConfig
from evacuation_simulation.simulation.generation import (
    generate_candidate_exits,
    generate_exit_configurations,
)
from evacuation_simulation.simulation.validator import make_validator


def sep(char="=", width=70):
    print(char * width)


def run_demo():
    sep()
    print(" STAGE 11 — REGULATORY & EXIT-PLACEMENT CONSTRAINT VALIDATION DEMO")
    sep()
    print()
    print("DISCLAIMER: Research/design-assistance tool only.")
    print("NOT legal certification. Results mean 'compliant with encoded")
    print("constraints in the selected regulation profile (NBC 2016 Part 4)'.")
    print()

    validator = make_validator()
    scenarios_dir = Path("scenarios")

    for scenario_name in ["basic_building", "bottleneck"]:
        filepath = scenarios_dir / (scenario_name + ".json")
        if not filepath.exists():
            print(f"[SKIP] {filepath} not found")
            continue

        cfg = SimulationConfig.from_json(filepath)
        fp = cfg.floor_plan
        occupants = cfg.occupants

        sep()
        print(f" SCENARIO: {cfg.scenario_name} ({filepath.name})")
        sep()
        print(f"  Grid: {fp.grid.rows}x{fp.grid.cols}")
        print(f"  Walls: {len(fp.walls)} cells")
        print(f"  DoorOpenings defined: {len(fp.doors)}")
        print()

        # 1. Candidate exterior doors
        ext_doors = [d for d in fp.doors if d.exterior]
        int_doors = [d for d in fp.doors if not d.exterior]
        print(f"  Exterior DoorOpenings ({len(ext_doors)}) — eligible as exits:")
        for d in ext_doors:
            label = f" [{d.door_id}]" if d.door_id else ""
            print(f"    + position={d.position}, width={d.width:.1f}m{label}")

        if int_doors:
            print(f"  Interior DoorOpenings ({len(int_doors)}) — INELIGIBLE as exits:")
            for d in int_doors:
                label = f" [{d.door_id}]" if d.door_id else ""
                print(f"    - position={d.position}, width={d.width:.1f}m{label}  [interior, excluded]")

        # 2. Candidate exits from validator
        candidates = validator.candidate_exits(fp)
        print()
        print(f"  Candidate emergency exits derived: {len(candidates)}")
        for c in candidates:
            label = f" [{c.door_id}]" if c.door_id else ""
            print(f"    * {c.position}, width={c.width:.1f}m{label}")

        if len(candidates) < 2:
            print("  [Insufficient candidates for further analysis]")
            print()
            continue

        # 3. VALID configuration (well-separated exits)
        configs = list(generate_exit_configurations(candidates, exit_count=2))

        # Find the configuration with maximum exit separation
        import math
        best_cfg = max(
            configs,
            key=lambda ec: math.sqrt(
                (ec.exits[0][0] - ec.exits[1][0]) ** 2 +
                (ec.exits[0][1] - ec.exits[1][1]) ** 2
            )
        )

        print()
        sep("-")
        print("  VALID CONFIGURATION CHECK")
        sep("-")
        exits_str = ", ".join(
            f"{pos} (w={w:.1f}m)" for pos, w in zip(best_cfg.exits, best_cfg.widths)
        )
        print(f"  Selected exits: {exits_str}")
        result = validator.check(fp, best_cfg, occupants)
        if result.is_compliant:
            print("  Compliance: PASS")
            print("  Violations: none")
        else:
            print("  Compliance: FAIL")
            for v in result.violations:
                print(f"    [VIOLATION] {v.rule_id}: {v.message}")
                if v.measured_value is not None:
                    print(f"      Measured={v.measured_value}{v.unit} Required={v.required_value}{v.unit}")

        # 4. INVALID configuration (interior door + narrow width)
        print()
        sep("-")
        print("  INVALID CONFIGURATION CHECK")
        sep("-")
        # Use interior door (if available) or closest exits
        if int_doors:
            bad_exits = [int_doors[0].position]
            bad_widths = [int_doors[0].width]
            reason = "interior door selected"
        else:
            # Use same exit twice (duplicate) to trigger R1
            bad_exits = [candidates[0].position, candidates[0].position]
            bad_widths = [0.3, 0.3]
            reason = "duplicate exit + narrow width"

        from evacuation_simulation.simulation.config import ExitConfiguration
        bad_config = ExitConfiguration(
            configuration_id="invalid_demo",
            exits=bad_exits,
            widths=bad_widths,
        )
        bad_exits_str = ", ".join(
            f"{pos} (w={w:.1f}m)" for pos, w in zip(bad_config.exits, bad_config.widths)
        )
        print(f"  Selected exits: {bad_exits_str} [{reason}]")
        bad_result = validator.check(fp, bad_config, occupants)
        if bad_result.is_compliant:
            print("  Compliance: PASS (unexpected)")
        else:
            print("  Compliance: FAIL (expected)")
            for v in bad_result.violations:
                print(f"    [VIOLATION] {v.rule_id}")
                print(f"      Message : {v.message[:100]}")
                if v.measured_value is not None:
                    print(f"      Measured: {v.measured_value} {v.unit}")
                    print(f"      Required: {v.required_value} {v.unit}")
                print(f"      Clause  : {v.clause}")
                print(f"      Severity: {v.severity}")
                if v.relevant_ids:
                    print(f"      Relevant: {v.relevant_ids}")
                print()

        # 5. Machine-readable output
        print()
        sep("-")
        print("  MACHINE-READABLE VALIDATION RESULT (JSON excerpt)")
        sep("-")
        result_dict = bad_result.model_dump()
        print(f"  is_compliant: {result_dict['is_compliant']}")
        print(f"  violations  : {len(result_dict['violations'])} item(s)")
        if result_dict["violations"]:
            v0 = result_dict["violations"][0]
            print(f"  violations[0]:")
            print(f"    rule_id       : {v0['rule_id']}")
            print(f"    severity      : {v0['severity']}")
            print(f"    measured_value: {v0['measured_value']}")
            print(f"    required_value: {v0['required_value']}")
            print(f"    unit          : {v0['unit']!r}")

        # 6. Action mask
        print()
        sep("-")
        print("  ACTION MASK (partial selection — first exit already chosen)")
        sep("-")
        from evacuation_simulation.simulation.config import ExitConfiguration as EC
        partial = EC(exits=[candidates[0].position], widths=[candidates[0].width])
        mask = validator.mask(fp, occupants, partial, candidates)
        for pos, allowed in sorted(mask.items()):
            status = "ALLOWED  " if allowed else "BLOCKED  "
            label = f" [{next((c.door_id for c in candidates if (c.position[0], c.position[1]) == pos), '')}]"
            note = " (already selected)" if pos == (candidates[0].position[0], candidates[0].position[1]) else ""
            print(f"    {status} {pos}{label}{note}")
        print()
        print("  NOTE: Mask only evaluates R1 (valid exterior door).")
        print("  R2/R3/R4/R5 require complete configuration — use check() for final validation.")
        print()

    sep()
    print(" REGULATORY PROFILE SUMMARY")
    sep()
    rs = validator.regulatory_summary()
    for r in rs["rules"]:
        print(f"  {r['rule_id']}: {r['description'][:70]}")
    print()
    print("  Verified values (from NBC 2016 Part 4):")
    print("    R5: 22.5m travel distance (RESIDENTIAL) .... VERIFIED")
    print("    R5: 30.0m travel distance (BUSINESS) ....... VERIFIED")
    print()
    print("  Unverified/configurable values:")
    print("    R2: min 2 exits (Section 4.4.2.4.3) ........ UNVERIFIED (conditional)")
    print("    R3: 1/2 diagonal non-sprinklered ........... UNVERIFIED (IBC formulation)")
    print("    R3: 1/3 diagonal sprinklered ............... UNVERIFIED (IBC formulation)")
    print("    R4: 1.0m absolute minimum width ............ UNVERIFIED (conditional Table 4)")
    print("    R4: width-per-person ........................ UNVERIFIED (null)")
    print("    R5: cell_size_m = 1.0m ..................... CONFIGURABLE (modelling assumption)")
    print("    R5: DEFAULT travel limit = 22.5m ........... UNVERIFIED (conservative default)")
    print()
    sep()
    print(" END OF STAGE 11 DEMO")
    sep()


if __name__ == "__main__":
    run_demo()
