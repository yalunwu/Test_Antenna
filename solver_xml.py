"""Export grid-aligned sources and PEC sheets without losing float precision."""
from pathlib import Path
import xml.etree.ElementTree as ET


def critical_property_names(path):
    """Properties whose unused geometry would invalidate the antenna run."""
    properties = ET.parse(path).find(".//Properties")
    return {node.get("Name") for node in properties if node.tag in ("Metal", "Excitation")}


def write_solver_xml(fdtd, csx, path):
    """Preserve double coordinates across the standalone solver's XML input.

    The bundled CSXCAD writer uses six decimal places in scientific notation
    for Box coordinates, but more digits for grid lines. A zero-width source
    or sheet can consequently miss its grid plane and be silently unused.
    Write both from their original doubles with round-trip precision.
    """
    path = Path(path)
    fdtd.Write2XML(str(path))
    tree = ET.parse(path)
    grid = tree.find(".//RectilinearGrid")
    for axis in "xyz":
        grid.find(axis.upper() + "Lines").text = ",".join(
            format(float(value), ".17g") for value in csx.GetGrid().GetLines(axis)
        )

    properties = {str(prop.GetID()): prop for prop in csx.GetAllProperties()}
    for element in tree.find(".//Properties"):
        primitives = properties[element.get("ID")].GetAllPrimitives()
        exported = list(element.find("Primitives"))
        if len(primitives) != len(exported):
            raise ValueError(f"Primitive count changed while exporting {element.get('Name')}")
        for primitive, node in zip(primitives, exported):
            if primitive.GetTypeName() != node.tag:
                raise ValueError(f"Primitive order changed while exporting {element.get('Name')}")
            if node.tag not in ("Box", "Cylinder"):
                continue
            for label, coordinates in (("P1", primitive.GetStart()), ("P2", primitive.GetStop())):
                for axis, value in zip("XYZ", coordinates):
                    node.find(label).set(axis, format(float(value), ".17g"))
    tree.write(path, encoding="utf-8", xml_declaration=True)
