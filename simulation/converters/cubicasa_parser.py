"""
cubicasa_parser.py
------------------
Stage 15A: Parsing and geometric extraction for CubiCasa5K SVG floor plans.

Extracts structured geometric primitives (walls, spaces, doors, columns)
and derives physical scale (pixels-per-meter) from available dimension annotations.
"""

from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

Point2D = tuple[float, float]
Polygon2D = list[Point2D]

SVG_NS = "{http://www.w3.org/2000/svg}"


class ScaleDerivationError(ValueError):
    """Raised when physical scale cannot be derived reliably from SVG annotations."""
    pass


@dataclass
class ParsedSpace:
    """A classified spatial boundary polygon with associated labels."""
    space_class: str
    name_label: str
    polygon: Polygon2D
    dimension_texts: list[str] = field(default_factory=list)
    dimension_marks: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class ParsedDoor:
    """A physical door opening threshold and its parent wall context."""
    door_id: str
    door_class: str
    parent_wall_class: str
    polygon: Polygon2D


@dataclass
class ParsedWall:
    """A structural wall barrier polygon."""
    wall_class: str
    polygon: Polygon2D
    doors: list[ParsedDoor] = field(default_factory=list)


@dataclass
class ParsedColumn:
    """A solid structural column / pillar."""
    column_class: str
    polygon: Polygon2D


@dataclass
class ParsedFixedFurniture:
    """A fixed furniture fixture (cabinets, sanitaryware, etc.)."""
    furniture_class: str
    polygon: Polygon2D


@dataclass
class ScaleInfo:
    """
    Physical scale derived from floor plan dimension annotations.

    Attributes
    ----------
    scale_px_per_m:
        Derived or specified ratio of SVG coordinate units (pixels) to 1.0 meter.
    is_derived:
        True if derived from SVG dimension annotations, False if manually specified.
    sample_count:
        Number of distinct dimension measurements used in derivation.
    median_scale:
        Median of derived measurements.
    min_scale:
        Minimum derived measurement.
    max_scale:
        Maximum derived measurement.
    std_dev:
        Standard deviation across measurements.
    """
    scale_px_per_m: float
    is_derived: bool
    sample_count: int = 0
    median_scale: Optional[float] = None
    min_scale: Optional[float] = None
    max_scale: Optional[float] = None
    std_dev: Optional[float] = None


@dataclass
class CubiCasaSVGData:
    """Container for raw parsed SVG elements."""
    viewbox_width: float
    viewbox_height: float
    spaces: list[ParsedSpace] = field(default_factory=list)
    walls: list[ParsedWall] = field(default_factory=list)
    doors: list[ParsedDoor] = field(default_factory=list)
    columns: list[ParsedColumn] = field(default_factory=list)
    furniture: list[ParsedFixedFurniture] = field(default_factory=list)


def parse_polygon_points(points_str: str) -> Polygon2D:
    """
    Parse an SVG polygon points attribute string into a list of (x, y) tuples.
    Handles both comma-separated and space-separated formats.
    """
    if not points_str or not points_str.strip():
        return []
    tokens = points_str.strip().replace(",", " ").split()
    points: Polygon2D = []
    for i in range(0, len(tokens) - 1, 2):
        try:
            x = float(tokens[i])
            y = float(tokens[i + 1])
            points.append((x, y))
        except ValueError:
            continue
    return points


def _parse_matrix_translation(transform_str: str) -> Optional[tuple[float, float]]:
    """Extract (tx, ty) from an SVG transform matrix(a,b,c,d,tx,ty)."""
    if not transform_str or "matrix(" not in transform_str:
        return None
    inner = transform_str.replace("matrix(", "").replace(")", "").strip()
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) == 6:
        try:
            return float(parts[4]), float(parts[5])
        except ValueError:
            return None
    return None


class CubiCasaParser:
    """
    Parser for CubiCasa5K model.svg files.
    """

    @classmethod
    def parse_file(cls, path: str | Path) -> CubiCasaSVGData:
        """
        Parse a model.svg file into structured CubiCasaSVGData.

        Raises
        ------
        FileNotFoundError:
            If path does not exist.
        ValueError:
            If the SVG file is malformed or lacks a valid viewBox.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"SVG floor-plan file not found: {p}")

        try:
            tree = ET.parse(p)
            root = tree.getroot()
        except ET.ParseError as err:
            raise ValueError(f"Malformed SVG XML in file {p}: {err}") from err

        return cls.parse_element(root)

    @classmethod
    def parse_element(cls, root: ET.Element) -> CubiCasaSVGData:
        """Parse from an ElementTree root element."""
        # Extract viewBox
        viewbox_str = root.attrib.get("viewBox", "")
        if not viewbox_str:
            # Fall back to width and height attributes
            w_str = root.attrib.get("width", "")
            h_str = root.attrib.get("height", "")
            if not w_str or not h_str:
                raise ValueError("SVG root lacks viewBox, width, and height attributes.")
            vb_w = float(w_str)
            vb_h = float(h_str)
        else:
            vb_parts = viewbox_str.strip().split()
            if len(vb_parts) < 4:
                raise ValueError(f"Invalid SVG viewBox attribute: '{viewbox_str}'")
            vb_w = float(vb_parts[2])
            vb_h = float(vb_parts[3])

        svg_data = CubiCasaSVGData(viewbox_width=vb_w, viewbox_height=vb_h)

        # 1. Parse Spaces
        for elem in root.iter(f"{SVG_NS}g"):
            cls_name = elem.attrib.get("class", "")
            if "Space " in cls_name or cls_name == "Space":
                poly_elem = elem.find(f"{SVG_NS}polygon")
                if poly_elem is None:
                    continue
                pts = parse_polygon_points(poly_elem.attrib.get("points", ""))
                if not pts:
                    continue

                # Room name
                name_elem = elem.find(f".//{SVG_NS}g[@id='NameLabel']/{SVG_NS}text")
                name_text = name_elem.text.strip() if name_elem is not None and name_elem.text else ""

                # Dimension text labels
                dim_texts: list[str] = []
                dim_label_elem = elem.find(f".//{SVG_NS}g[@id='DimensionMeasureLabel']")
                if dim_label_elem is not None:
                    for t in dim_label_elem.findall(f"{SVG_NS}text"):
                        if t.text:
                            dim_texts.append(t.text.strip())

                # Dimension marks
                dim_marks: dict[str, tuple[float, float]] = {}
                for mark in elem.findall(f".//{SVG_NS}g[@class='DimensionMark']"):
                    m_id = mark.attrib.get("id", "")
                    tr = mark.attrib.get("transform", "")
                    trans = _parse_matrix_translation(tr)
                    if m_id and trans is not None:
                        dim_marks[m_id] = trans

                svg_data.spaces.append(
                    ParsedSpace(
                        space_class=cls_name,
                        name_label=name_text,
                        polygon=pts,
                        dimension_texts=dim_texts,
                        dimension_marks=dim_marks,
                    )
                )

        # 2. Parse Walls and nested Doors
        door_counter = 1
        for elem in root.iter(f"{SVG_NS}g"):
            elem_id = elem.attrib.get("id", "")
            if elem_id == "Wall":
                wall_cls = elem.attrib.get("class", "Wall")
                poly_elem = elem.find(f"{SVG_NS}polygon")
                if poly_elem is None:
                    continue
                w_pts = parse_polygon_points(poly_elem.attrib.get("points", ""))
                if not w_pts:
                    continue

                parsed_wall = ParsedWall(wall_class=wall_cls, polygon=w_pts)

                # Search for nested Door elements inside this wall
                for d_elem in elem.findall(f"{SVG_NS}g"):
                    if d_elem.attrib.get("id") == "Door":
                        d_cls = d_elem.attrib.get("class", "Door")
                        d_poly = d_elem.find(f"{SVG_NS}polygon")
                        if d_poly is None:
                            continue
                        d_pts = parse_polygon_points(d_poly.attrib.get("points", ""))
                        if not d_pts:
                            continue

                        door_id = f"door_{door_counter}"
                        door_counter += 1
                        parsed_door = ParsedDoor(
                            door_id=door_id,
                            door_class=d_cls,
                            parent_wall_class=wall_cls,
                            polygon=d_pts,
                        )
                        parsed_wall.doors.append(parsed_door)
                        svg_data.doors.append(parsed_door)

                svg_data.walls.append(parsed_wall)

        # 3. Parse Columns
        for elem in root.iter(f"{SVG_NS}g"):
            if elem.attrib.get("id") == "Column":
                col_cls = elem.attrib.get("class", "Column")
                c_poly = elem.find(f"{SVG_NS}polygon")
                if c_poly is not None:
                    c_pts = parse_polygon_points(c_poly.attrib.get("points", ""))
                    if c_pts:
                        svg_data.columns.append(ParsedColumn(column_class=col_cls, polygon=c_pts))

        # 4. Parse Fixed Furniture
        for elem in root.iter(f"{SVG_NS}g"):
            cls_name = elem.attrib.get("class", "")
            if "FixedFurniture " in cls_name or cls_name == "FixedFurniture":
                f_poly = elem.find(f".//{SVG_NS}polygon")
                if f_poly is not None:
                    f_pts = parse_polygon_points(f_poly.attrib.get("points", ""))
                    if f_pts:
                        svg_data.furniture.append(
                            ParsedFixedFurniture(furniture_class=cls_name, polygon=f_pts)
                        )

        return svg_data


def parse_cubicasa_svg(path: str | Path) -> CubiCasaSVGData:
    """Convenience functional wrapper for CubiCasaParser.parse_file."""
    return CubiCasaParser.parse_file(path)


class ScaleEstimator:
    """
    Derives physical scale (pixels-per-meter) from ParsedSpace dimensions.
    """

    @classmethod
    def estimate_scale(
        cls,
        spaces: list[ParsedSpace],
        manual_override: Optional[float] = None,
        max_relative_std_dev: float = 0.05,
    ) -> ScaleInfo:
        """
        Estimate the SVG scale in pixels per meter.

        Parameters
        ----------
        spaces:
            List of parsed spaces from CubiCasaParser.
        manual_override:
            Optional explicit scale in pixels per meter. If provided, bypasses derivation.
        max_relative_std_dev:
            Maximum allowed coefficient of variation (std_dev / median) before
            flagging derivation as unreliable. Default is 5% (0.05).

        Returns
        -------
        ScaleInfo
            Structured scale information.

        Raises
        ------
        ScaleDerivationError:
            If manual_override is None and usable dimension data is missing or inconsistent.
        """
        if manual_override is not None:
            if manual_override <= 0.0:
                raise ScaleDerivationError(f"Manual scale must be positive, got {manual_override}.")
            return ScaleInfo(scale_px_per_m=manual_override, is_derived=False, sample_count=0)

        ratios: list[float] = []

        for space in spaces:
            # Look for mark2 (width) and mark3 (height)
            w_px = space.dimension_marks.get("mark2", (None, None))[0]
            h_px = space.dimension_marks.get("mark3", (None, None))[1]

            # Look for metric text: e.g. "3.23 m x 2.28 m"
            metric_text: Optional[str] = None
            for txt in space.dimension_texts:
                if "m" in txt and "x" in txt:
                    metric_text = txt
                    break

            if not metric_text:
                continue

            try:
                parts = metric_text.replace("m", "").split("x")
                if len(parts) != 2:
                    continue
                w_m = float(parts[0].strip())
                h_m = float(parts[1].strip())
            except ValueError:
                continue

            if w_px is not None and w_m > 0:
                ratios.append(w_px / w_m)
            if h_px is not None and h_m > 0:
                ratios.append(h_px / h_m)

        if not ratios:
            raise ScaleDerivationError(
                "Unable to derive physical scale: no usable metric dimension labels "
                "with corresponding dimension marks found. Manual scale calibration is required."
            )

        med = statistics.median(ratios)
        std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        cv = std / med if med > 0 else float("inf")

        if cv > max_relative_std_dev:
            raise ScaleDerivationError(
                f"Derived scale is inconsistent across annotated spaces (coefficient of variation: "
                f"{cv:.2%}, expected < {max_relative_std_dev:.1%}). "
                f"Manual calibration required."
            )

        return ScaleInfo(
            scale_px_per_m=med,
            is_derived=True,
            sample_count=len(ratios),
            median_scale=med,
            min_scale=min(ratios),
            max_scale=max(ratios),
            std_dev=std,
        )
