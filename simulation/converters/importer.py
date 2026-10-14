"""
importer.py
-----------
Stage 15A: End-to-end CubiCasa5K Floor Plan Importer.

Coordinates SVG parsing, scale estimation, door classification, grid discretization,
and Pydantic FloorPlan assembly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from evacuation_simulation.simulation.config import (
    ExitConfiguration,
    FloorPlan,
    GridConfig,
    OccupantScenario,
    SimulationConfig,
    SimulationParameters,
)
from evacuation_simulation.simulation.converters.cubicasa_parser import (
    CubiCasaParser,
    CubiCasaSVGData,
    ScaleEstimator,
    ScaleInfo,
)
from evacuation_simulation.simulation.converters.door_classifier import (
    DoorClassifier,
    DoorReviewRecord,
)
from evacuation_simulation.simulation.converters.grid_discretizer import (
    DiscretizationResult,
    GridDiscretizer,
)
from evacuation_simulation.simulation.generation import (
    CandidateExit,
    generate_candidate_exits,
)


@dataclass
class CubiCasaImportResult:
    """Complete result of importing a CubiCasa5K floor plan."""
    floor_plan: FloorPlan
    scale_info: ScaleInfo
    candidate_exits: list[CandidateExit]
    door_records: list[DoorReviewRecord]
    review_manifest: dict[str, Any]
    discretization: DiscretizationResult
    cell_size_m: float
    source_path: str

    def export_review_manifest(self, output_path: str | Path) -> None:
        """Write the review manifest to a formatted JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.review_manifest, fh, indent=2)

    def export_scenario_json(
        self,
        output_path: str | Path,
        scenario_name: Optional[str] = None,
        max_timesteps: int = 500,
        random_seed: int = 42,
    ) -> Path:
        """
        Export as a valid PATH_RL SimulationConfig scenario JSON.
        Uses candidate exits as initial ExitConfiguration.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        name = scenario_name or Path(self.source_path).parent.name or "cubicasa_import"

        # Construct exit configuration from justified candidate exits
        exits = [list(c.position) for c in self.candidate_exits]
        widths = [c.width for c in self.candidate_exits]

        # If no candidate exits were justified, provide first door as reference to allow schema validation
        if not exits and self.floor_plan.doors:
            first_d = self.floor_plan.doors[0]
            exits = [list(first_d.position)]
            widths = [first_d.width]

        exit_cfg = ExitConfiguration(
            configuration_id="reference",
            exits=exits,
            widths=widths,
        )

        sim_config = SimulationConfig(
            scenario_name=f"cubicasa_{name}",
            floor_plan=self.floor_plan,
            exit_configuration=exit_cfg,
            occupants=OccupantScenario(agents=[]),
            parameters=SimulationParameters(
                max_timesteps=max_timesteps,
                random_seed=random_seed,
            ),
        )

        with p.open("w", encoding="utf-8") as fh:
            json.dump(sim_config.model_dump(), fh, indent=2)

        return p


def import_cubicasa_plan(
    svg_path: str | Path,
    cell_size_m: float = 0.5,
    manual_scale_px_per_m: Optional[float] = None,
) -> CubiCasaImportResult:
    """
    Import a CubiCasa5K model.svg floor plan into a validated PATH_RL FloorPlan.

    Parameters
    ----------
    svg_path:
        Path to the CubiCasa5K model.svg file.
    cell_size_m:
        Physical size of one grid cell in meters (default 0.5m).
    manual_scale_px_per_m:
        Optional manual override for pixels per meter. If None, derived from
        dimension annotations.

    Returns
    -------
    CubiCasaImportResult
        Validated FloorPlan, candidate exits, scale info, and review manifest.

    Raises
    ------
    FileNotFoundError:
        If svg_path does not exist.
    ScaleDerivationError:
        If scale cannot be derived and manual_scale_px_per_m is None.
    ValueError:
        If SVG is malformed or discretization produces invalid geometry.
    pydantic.ValidationError:
        If assembled FloorPlan violates PATH_RL constraints.
    """
    path_str = str(svg_path)

    # 1. Parse SVG elements
    svg_data = CubiCasaParser.parse_file(svg_path)

    # 2. Derive or apply physical scale
    scale_info = ScaleEstimator.estimate_scale(
        spaces=svg_data.spaces,
        manual_override=manual_scale_px_per_m,
    )

    # 3. Classify doors
    door_records = DoorClassifier.classify(
        doors=svg_data.doors,
        spaces=svg_data.spaces,
        scale_px_per_m=scale_info.scale_px_per_m,
    )

    # 4. Continuous-to-grid discretization
    discretization = GridDiscretizer.discretize(
        svg_data=svg_data,
        door_records=door_records,
        scale_px_per_m=scale_info.scale_px_per_m,
        cell_size_m=cell_size_m,
    )

    # 5. Assemble and validate FloorPlan model
    floor_plan = FloorPlan(
        grid=GridConfig(rows=discretization.grid_rows, cols=discretization.grid_cols),
        walls=discretization.walls,
        doors=discretization.doors,
    )

    # 6. Generate candidate exits from justified exterior openings
    candidate_exits = generate_candidate_exits(floor_plan, fallback_legacy_boundary=False)

    # 7. Construct review manifest
    manifest: dict[str, Any] = {
        "source_file": path_str,
        "grid_resolution_m": cell_size_m,
        "grid_shape": [discretization.grid_rows, discretization.grid_cols],
        "scale_px_per_meter": round(scale_info.scale_px_per_m, 2),
        "scale_is_derived": scale_info.is_derived,
        "scale_sample_count": scale_info.sample_count,
        "scale_std_dev": round(scale_info.std_dev, 4) if scale_info.std_dev is not None else None,
        "wall_cell_count": len(discretization.walls),
        "total_doors_detected": len(discretization.doors),
        "candidate_exterior_exits_derived": len(candidate_exits),
        "doors": [d.to_dict() for d in door_records],
    }

    return CubiCasaImportResult(
        floor_plan=floor_plan,
        scale_info=scale_info,
        candidate_exits=candidate_exits,
        door_records=door_records,
        review_manifest=manifest,
        discretization=discretization,
        cell_size_m=cell_size_m,
        source_path=path_str,
    )
