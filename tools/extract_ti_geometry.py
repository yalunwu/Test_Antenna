"""Rebuild TI outline JSON from the archived Gerbers (dev dependency: shapely==2.1.2).

These two legacy files use inch units, 2.4 coordinates, circular apertures,
linear strokes, and filled regions. Unsupported commands fail explicitly.
Shapely is only needed for this extraction, not for simulation.
"""
from pathlib import Path
import hashlib
import json
import re

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.affinity import translate

ROOT = Path(__file__).resolve().parents[1]
SCALE = 0.0001 * 25.4


def read_gerber(path):
    apertures, shapes = {}, []
    position, aperture, operation, region = None, None, 2, None
    text = path.read_text(encoding="ascii")
    if "G70*" not in text or "%FSLAX24Y24*%" not in text:
        raise ValueError("Expected inch, absolute 2.4 Gerber coordinates")
    for token in text.split("*"):
        token = token.replace("%", "").strip()
        if not token:
            continue
        match = re.fullmatch(r"ADD(\d+)C,([\d.]+)", token)
        if match:
            apertures[int(match[1])] = float(match[2]) * 25.4
            continue
        if token == "G36":
            region = []
            continue
        if token == "G37":
            polygon = Polygon(region)
            if not polygon.is_valid:
                raise ValueError(f"Invalid Gerber region in {path}")
            shapes.append(polygon)
            region = None
            continue
        if re.fullmatch(r"D\d+", token):
            code = int(token[1:])
            if code >= 10:
                aperture = code
            else:
                operation = code
            continue
        if token.startswith(("FS", "G70", "M02")):
            continue
        match = re.fullmatch(r"(?:X(-?\d+))?(?:Y(-?\d+))?(?:D0?([123]))?", token)
        if not match or not (match[1] or match[2]):
            raise ValueError(f"Unsupported Gerber command: {token}")
        x = float(match[1]) * SCALE if match[1] else position[0]
        y = float(match[2]) * SCALE if match[2] else position[1]
        if match[3]:
            operation = int(match[3])
        point = (x, y)
        if region is not None:
            region.append(point)
        elif operation == 1:
            shapes.append(LineString([position, point]).buffer(apertures[aperture] / 2, quad_segs=8))
        elif operation == 3:
            shapes.append(Point(point).buffer(apertures[aperture] / 2, quad_segs=8))
        position = point
    copper = unary_union(shapes)
    if not copper.is_valid:
        raise ValueError(f"Invalid merged copper in {path}")
    return copper


def main():
    # Origins are copper feed centers in X and the reference ground edge in Y.
    # AN043: D4=0.5 mm overlap ends at Gerber Y=30148 (Figure 3).
    # IFA: ground feet include the 8-mil outline stroke: Y=38060-40.
    cases = [("cc2511", "IFA_USB.spl", 20294.5, 30148, -98.5),
             ("cc2430", "Inverted_F_Antenna.spl", 18411, 38020, -10)]
    for directory, filename, x_origin, y_origin, feed_y in cases:
        path = ROOT / "docs_reference" / directory / filename
        copper = translate(read_gerber(path), -x_origin * SCALE, -y_origin * SCALE)
        polygons = [copper] if copper.geom_type == "Polygon" else list(copper.geoms)
        if any(p.interiors for p in polygons):
            raise ValueError("Outline has holes; explicit hole handling is required")
        data = {"source_file": filename, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "units": "mm", "source_coordinate_unit_mm": SCALE,
                "source_origin": [x_origin, y_origin], "feed_center_mm": [0, feed_y * SCALE],
                "bounds_mm": list(copper.bounds), "area_mm2": copper.area,
                "polygons_mm": [[[round(x, 8), round(y, 8)] for x, y in p.exterior.coords[:-1]]
                                for p in polygons],
                "notes": "Filled regions and circular-aperture strokes merged; arcs sampled at 32 segments/circle."}
        (path.parent / "antenna_outline.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(directory, "components", len(polygons), "bounds", copper.bounds, "vertices", sum(len(p.exterior.coords) for p in polygons))


if __name__ == "__main__":
    main()
