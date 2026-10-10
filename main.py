"""
Main Entry Point for 77 GHz 4-Channel Patch Antenna Array Simulation
Compatible with PyCharm, CLI, and interactive runs.
"""
import argparse
import sys
import pathlib

# Ensure project root is in path
project_root = pathlib.Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import openems_config
from simulate import run_simulation

def main():
    parser = argparse.ArgumentParser(
        description="77 GHz 4-Channel Patch Antenna Array Simulation using openEMS FDTD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--engine',
        choices=['vulkan', 'multithreaded'],
        default='vulkan',
        help="openEMS solver engine ('vulkan' GPU or 'multithreaded' CPU)"
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=0,
        help="Number of threads for multithreaded engine (0 = auto-detect)"
    )
    parser.add_argument(
        '--view',
        action='store_true',
        help="Open the 3D model in AppCSXCAD viewer after simulation setup"
    )
    parser.add_argument(
        '--post-process',
        type=str,
        default=None,
        help="Path to an existing timestamped directory in SimResult to re-run post-processing"
    )
    parser.add_argument(
        '--cells-per-wavelength',
        type=float,
        default=40.0,
        help="Target mesh resolution in cells per wavelength (e.g. 22=standard ~770k cells, 30=fine ~1.8M cells, 40=ultra ~4.1M cells)"
    )
    
    args = parser.parse_args()
    
    if args.post_process:
        from simulate import post_process_directory
        target_dir = pathlib.Path(args.post_process).resolve()
        post_process_directory(target_dir)
        output_dir = target_dir
    else:
        output_dir = run_simulation(
            engine=args.engine,
            num_threads=args.threads,
            cells_per_wavelength=args.cells_per_wavelength
        )
    
    if args.view:
        xml_file = output_dir / "openEMS_model.xml"
        openems_config.view_in_appcsxcad(xml_file)

if __name__ == '__main__':
    main()
