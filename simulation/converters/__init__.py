"""
converters
----------
Stage 15A: Floor Plan Import & Parsing Layer.

Provides modular converters from external floor-plan formats
(e.g., CubiCasa5K SVG) into validated PATH_RL FloorPlan representations.
"""

from evacuation_simulation.simulation.converters.cubicasa_parser import (
    CubiCasaParser,
    CubiCasaSVGData,
    ParsedDoor,
    ParsedSpace,
    ParsedWall,
    ScaleDerivationError,
    ScaleEstimator,
    ScaleInfo,
    parse_cubicasa_svg,
)
from evacuation_simulation.simulation.converters.door_classifier import (
    DoorClassification,
    DoorClassifier,
    DoorReviewRecord,
    classify_doors,
)
from evacuation_simulation.simulation.converters.grid_discretizer import (
    DiscretizationResult,
    GridDiscretizer,
    discretize_plan,
)
from evacuation_simulation.simulation.converters.importer import (
    CubiCasaImportResult,
    import_cubicasa_plan,
)

__all__ = [
    "CubiCasaParser",
    "CubiCasaSVGData",
    "ParsedDoor",
    "ParsedSpace",
    "ParsedWall",
    "ScaleDerivationError",
    "ScaleEstimator",
    "ScaleInfo",
    "parse_cubicasa_svg",
    "DoorClassification",
    "DoorClassifier",
    "DoorReviewRecord",
    "classify_doors",
    "DiscretizationResult",
    "GridDiscretizer",
    "discretize_plan",
    "CubiCasaImportResult",
    "import_cubicasa_plan",
]
