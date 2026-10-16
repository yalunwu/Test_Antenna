"""Export previews and check native solver setup without an RF simulation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import antenna_model
from openEMS import openEMS
from solver_xml import write_solver_xml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "SimResult" / "geometry_validation_v2")
    parser.add_argument("--native", action="store_true", help="Also construct the native Vulkan FDTD operators")
    parser.add_argument("--native-child", choices=list(antenna_model.SUPPORTED_ANTENNAS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.native_child:
        # Exercise exactly the serialized model consumed by openEMS.exe.
        fdtd = openEMS()
        if not fdtd.ReadFromXML(str(args.output / 'openEMS_model.xml')):
            raise RuntimeError('Could not reload the exported solver model')
        csx = fdtd.GetCSX()
        status = fdtd.Run(str(args.output.resolve()), setup_only=True, engine='vulkan',
                          vulkan_pml='fused', disable_dumps=True, dump_statistics=True)
        unused = []
        for primitive in csx.GetAllPrimitives():
            prop = primitive.GetProperty()
            if prop.GetTypeString() in ('Metal', 'Excitation') and not primitive.GetPrimitiveUsed():
                item = {'type': primitive.GetTypeName(), 'id': primitive.GetID(),
                        'priority': primitive.GetPriority(), 'property': prop.GetName()}
                if primitive.GetTypeName() == 'Box':
                    item.update(start=primitive.GetStart().tolist(), stop=primitive.GetStop().tolist())
                unused.append(item)
        (args.output.resolve() / 'unused_critical_primitives.json').write_text(json.dumps(unused, indent=2))
        if unused:
            raise RuntimeError(f'{len(unused)} metal/source primitives are absent from the native discretization')
        print('NATIVE_SETUP_RETURN', status, flush=True)
        sys.exit(status)
    records = []
    for name in antenna_model.SUPPORTED_ANTENNAS:
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        model = antenna_model.get_antenna_model(name)
        fdtd, csx, ports, nf = model.create_simulation()
        errors = csx.Update()
        if errors:
            raise RuntimeError(errors)
        xml = directory / "openEMS_model.xml"
        write_solver_xml(fdtd, csx, xml.resolve())
        model.draw_layout(str(directory / "antenna_layout.png"))
        model.draw_3d_view(str(directory / "antenna_3d_view.png"))
        model.draw_interactive_3d(str(directory / "antenna_3d_interactive.html"))
        record = {"name": name, "geometry_revision": getattr(model, "geometry_revision", "legacy-v1"),
                  "grid_cells": int(np.prod([len(csx.GetGrid().GetLines(axis)) for axis in "xyz"])),
                  "rf_ground_depth_mm": getattr(model, "rf_ground_depth", None),
                  "source_sha256": getattr(model, "reference_outline", {}).get("source_sha256")}
        if args.native:
            command = [sys.executable, str(Path(__file__).resolve()), '--native-child', name,
                       '--output', str(directory.resolve())]
            with (directory / "native_setup.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{name}: native setup failed ({result.returncode}); see {directory}")
            record["native_setup_exit_code"] = result.returncode
            record["validated_input"] = "reloaded_solver_xml"
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "validation_summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
