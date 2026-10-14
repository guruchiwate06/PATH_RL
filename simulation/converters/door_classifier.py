"""
door_classifier.py
------------------
Stage 15A: Topological & Contextual Door Classification.

Evaluates each parsed door opening based on geometry, parent wall classification,
and adjacent space semantics.

Produces a human-readable and machine-readable JSON review manifest.
Strictly distinguishes justified exterior openings from balcony doors,
interior passages, and uncertain cases.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from evacuation_simulation.simulation.converters.cubicasa_parser import (
    ParsedDoor,
    ParsedSpace,
    Point2D,
    Polygon2D,
)


class DoorClassification(str, Enum):
    """Semantic classification of a door opening."""
    INTERIOR = "interior"
    POTENTIAL_EXTERIOR = "potential_exterior"
    BALCONY_OUTDOOR = "balcony_outdoor"
    UNCERTAIN = "uncertain"


@dataclass
class DoorReviewRecord:
    """
    Detailed classification and review record for a single door opening.
    """
    door_id: str
    svg_center: tuple[float, float]
    estimated_width_m: float
    parent_wall_class: str
    adjacent_spaces: list[str]
    classification: DoorClassification
    confidence: float
    review_required: bool
    review_reason: str
    is_candidate_exit: bool
    grid_position: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        d["classification"] = self.classification.value
        d["svg_center"] = [round(self.svg_center[0], 2), round(self.svg_center[1], 2)]
        d["estimated_width_m"] = round(self.estimated_width_m, 2)
        if self.grid_position is not None:
            d["grid_position"] = list(self.grid_position)
        return d


def point_in_polygon(x: float, y: float, poly: Polygon2D) -> bool:
    """Standard ray-casting point-in-polygon algorithm."""
    if not poly or len(poly) < 3:
        return False
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _find_adjacent_spaces(
    door_pts: Polygon2D,
    spaces: list[ParsedSpace],
) -> tuple[list[ParsedSpace], list[ParsedSpace]]:
    """
    Sample points slightly on either side of the door opening to determine adjacent spaces.
    """
    xs = [p[0] for p in door_pts]
    ys = [p[1] for p in door_pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    w_px = max(xs) - min(xs)
    h_px = max(ys) - min(ys)

    # Offset along the normal (thickness) direction
    if w_px > h_px:
        # Horizontal opening: sample above and below
        offset = max(h_px, 15.0)
        sample1 = (cx, cy - offset)
        sample2 = (cx, cy + offset)
    else:
        # Vertical opening: sample left and right
        offset = max(w_px, 15.0)
        sample1 = (cx - offset, cy)
        sample2 = (cx + offset, cy)

    side1 = [s for s in spaces if point_in_polygon(sample1[0], sample1[1], s.polygon)]
    side2 = [s for s in spaces if point_in_polygon(sample2[0], sample2[1], s.polygon)]
    return side1, side2


class DoorClassifier:
    """
    Classifier for door openings in CubiCasa5K floor plans.
    """

    @classmethod
    def classify(
        cls,
        doors: list[ParsedDoor],
        spaces: list[ParsedSpace],
        scale_px_per_m: float,
    ) -> list[DoorReviewRecord]:
        """
        Classify all doors and produce review records.

        Parameters
        ----------
        doors:
            Parsed doors from SVG.
        spaces:
            Parsed spaces from SVG.
        scale_px_per_m:
            Scale factor in pixels per meter.

        Returns
        -------
        list[DoorReviewRecord]
            Structured classification records.
        """
        records: list[DoorReviewRecord] = []

        for door in doors:
            pts = door.polygon
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            span_px = max(max(xs) - min(xs), max(ys) - min(ys))
            width_m = span_px / scale_px_per_m if scale_px_per_m > 0 else 1.0

            side1, side2 = _find_adjacent_spaces(pts, spaces)

            def space_label(sp_list: list[ParsedSpace]) -> str:
                if not sp_list:
                    return "Exterior/Void"
                s = sp_list[0]
                name = s.name_label or s.space_class.replace("Space ", "")
                return f"{name} ({s.space_class.replace('Space ', '')})"

            label1 = space_label(side1)
            label2 = space_label(side2)
            adj_spaces = [label1, label2]

            # Classification logic
            is_parent_external = "External" in door.parent_wall_class

            # Check if any adjacent space is an outdoor space/balcony
            has_outdoor = any(
                "Outdoor" in s.space_class or "ULKOTILA" in s.name_label
                for s in (side1 + side2)
            )

            # Check if any adjacent space is an entry or draught lobby
            has_entry = any(
                "Entry" in s.space_class
                or "DraughtLobby" in s.space_class
                or s.name_label in ("ET", "TK")
                for s in (side1 + side2)
            )

            if not is_parent_external:
                # Interior partition door
                records.append(
                    DoorReviewRecord(
                        door_id=door.door_id,
                        svg_center=(cx, cy),
                        estimated_width_m=width_m,
                        parent_wall_class=door.parent_wall_class,
                        adjacent_spaces=adj_spaces,
                        classification=DoorClassification.INTERIOR,
                        confidence=0.95,
                        review_required=False,
                        review_reason="Standard interior room-to-room or room-to-hallway passage.",
                        is_candidate_exit=False,
                    )
                )
            elif has_entry:
                # Potential exterior opening from draught lobby / entry hall
                records.append(
                    DoorReviewRecord(
                        door_id=door.door_id,
                        svg_center=(cx, cy),
                        estimated_width_m=width_m,
                        parent_wall_class=door.parent_wall_class,
                        adjacent_spaces=adj_spaces,
                        classification=DoorClassification.POTENTIAL_EXTERIOR,
                        confidence=0.75,
                        review_required=True,
                        review_reason=(
                            "Primary unit entrance opening connecting draught lobby (TK) to exterior porch/access route. "
                            "Justified candidate exit pending ground-level egress verification."
                        ),
                        is_candidate_exit=True,
                    )
                )
            elif has_outdoor:
                # Check if it opens from a living space (balcony) or utility space (storage)
                is_utility = any("Storage" in s.space_class or "VAR" in s.name_label for s in (side1 + side2))
                if is_utility:
                    records.append(
                        DoorReviewRecord(
                            door_id=door.door_id,
                            svg_center=(cx, cy),
                            estimated_width_m=width_m,
                            parent_wall_class=door.parent_wall_class,
                            adjacent_spaces=adj_spaces,
                            classification=DoorClassification.UNCERTAIN,
                            confidence=0.40,
                            review_required=True,
                            review_reason=(
                                "Opens from storage space to exterior/porch; "
                                "unknown egress feasibility or utility access only."
                            ),
                            is_candidate_exit=False,
                        )
                    )
                else:
                    # Door leads to private balcony / outdoor space from living room / bedroom
                    records.append(
                        DoorReviewRecord(
                            door_id=door.door_id,
                            svg_center=(cx, cy),
                            estimated_width_m=width_m,
                            parent_wall_class=door.parent_wall_class,
                            adjacent_spaces=adj_spaces,
                            classification=DoorClassification.BALCONY_OUTDOOR,
                            confidence=0.90,
                            review_required=True,
                            review_reason=(
                                "Opens from living space to private balcony/outdoor enclosure; "
                                "unverified egress to ground level or safe dispersal area."
                            ),
                            is_candidate_exit=False,
                        )
                    )
            else:
                # Exterior wall door from other space
                records.append(
                    DoorReviewRecord(
                        door_id=door.door_id,
                        svg_center=(cx, cy),
                        estimated_width_m=width_m,
                        parent_wall_class=door.parent_wall_class,
                        adjacent_spaces=adj_spaces,
                        classification=DoorClassification.UNCERTAIN,
                        confidence=0.30,
                        review_required=True,
                        review_reason=(
                            "Exterior wall door with ambiguous spatial context; "
                            "unknown external elevation or egress destination."
                        ),
                        is_candidate_exit=False,
                    )
                )

        return records


def classify_doors(
    doors: list[ParsedDoor],
    spaces: list[ParsedSpace],
    scale_px_per_m: float,
) -> list[DoorReviewRecord]:
    """Convenience functional wrapper for DoorClassifier.classify."""
    return DoorClassifier.classify(doors, spaces, scale_px_per_m)
