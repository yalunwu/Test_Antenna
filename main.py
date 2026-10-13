"""
Main Entry Point for Proven RF Antenna Simulations using openEMS FDTD.
Compatible with PyCharm, CLI, and interactive runs.

Supported Proven TI & Qorvo/Decawave Antennas:
  - 'ti_77ghz_rx'      : TI 77 GHz RX Series-Fed Patch Array (SWRU553A Section 2.6.2)
  - 'ti_24ghz_an043'    : TI 2.4 GHz Small Size MIFA (AN043 / SWRA117D Table 1)
  - 'ti_24ghz_swru120d' : TI 2.4 GHz Inverted-F Antenna (SWRU120D / DN0007 Table 1-1)
  - 'mdpi_24ghz_array'  : 24 GHz 1x4 Series-Fed Patch Array (MDPI Electronics 2023, 12, 1512)
  - 'mdpi_24ghz_plates' : 24 GHz 1x4 Patch Array with Parasitic Metal Plates (MDPI 2023)
  - 'qorvo_wb002'       : Qorvo/Decawave WB002 UWB 3.0-8.0 GHz Planar Monopole MAN Reference Antenna
  - 'all'               : Run simulation suite across all antennas sequentially
"""
import argparse
import sys
import pathlib

# Ensure project root is in path
project_root = pathlib.Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import openems_config
import antenna_model
from simulate import run_simulation, post_process_directory

def main():
    parser = argparse.ArgumentParser(
        description="Proven TI & Qorvo RF Antenna openEMS FDTD Electromagnetic Simulation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--antenna',
        choices=['ti_77ghz_rx', 'ti_24ghz_an043', 'ti_24ghz_swru120d', 'mdpi_24ghz_array', 'mdpi_24ghz_plates', 'qorvo_wb002', 'all'],
        default='ti_77ghz_rx',
        help="Antenna model to simulate: 'ti_77ghz_rx', 'ti_24ghz_an043', 'ti_24ghz_swru120d', 'mdpi_24ghz_array', 'mdpi_24ghz_plates', 'qorvo_wb002', or 'all'"
    )
    parser.add_argument(
        '--engine',
        choices=['vulkan', 'multithreaded'],
        default='vulkan',
        help="openEMS solver engine ('vulkan' GPU acceleration or 'multithreaded' CPU)"
    )
    parser.add_argument(
        '--vulkan-pml',
        choices=['indexed', 'fused'],
        default='fused',
        help='Vulkan PML stepping mode'
    )
    parser.add_argument(
        '--vulkan-profile',
        action='store_true',
        help="Log Vulkan CPU/GPU execution timings and memory counters"
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=0,
        help="Number of threads for CPU solver (0 = auto-detect)"
    )
    parser.add_argument(
        '--view',
        action='store_true',
        help="Open the 3D geometry in AppCSXCAD viewer after simulation setup"
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
        default=60.0,
        help="Target mesh resolution in cells per wavelength (default: 60.0, minimum: >= 60.0)"
    )
    parser.add_argument(
        '--channel',
        choices=['1', '2', '3', '4', 'all'],
        default='1',
        help="Active channel excitation for multi-channel arrays (ignored for single-channel models)"
    )
    parser.add_argument(
        '--max-timesteps',
        type=int,
        default=40000,
        help="Maximum FDTD solver timesteps"
    )
    
    args = parser.parse_args()
    if args.cells_per_wavelength < 60.0:
        print(f"[Notice] Enforcing minimum mesh resolution: setting cells_per_wavelength from {args.cells_per_wavelength} to 60.0")
        args.cells_per_wavelength = 60.0
    if args.vulkan_profile and args.engine != 'vulkan':
        parser.error("--vulkan-profile requires --engine vulkan")
    
    if args.post_process:
        target_dir = pathlib.Path(args.post_process).resolve()
        # Detect antenna model from dir name if possible
        selected_ant = args.antenna
        for ant_key in antenna_model.SUPPORTED_ANTENNAS:
            if ant_key in target_dir.name.lower():
                selected_ant = ant_key
                break
        ant_instance = antenna_model.get_antenna_model(selected_ant, cells_per_wavelength=args.cells_per_wavelength)
        post_process_directory(
            target_dir,
            antenna=ant_instance,
            channel=args.channel,
            max_timesteps=args.max_timesteps
        )
        output_dirs = [target_dir]
    else:
        if args.antenna == 'all':
            antennas_to_run = ['ti_77ghz_rx', 'ti_24ghz_an043', 'ti_24ghz_swru120d', 'mdpi_24ghz_array', 'mdpi_24ghz_plates', 'qorvo_wb002']
        else:
            antennas_to_run = [args.antenna]
            
        output_dirs = []
        for ant_name in antennas_to_run:
            print(f"\n>>> Running simulation for {ant_name}...\n")
            out = run_simulation(
                antenna_type=ant_name,
                engine=args.engine,
                num_threads=args.threads,
                vulkan_profile=args.vulkan_profile,
                vulkan_pml=args.vulkan_pml,
                cells_per_wavelength=args.cells_per_wavelength,
                channel=args.channel,
                max_timesteps=args.max_timesteps
            )
            output_dirs.append(out)
            
    if args.view and output_dirs:
        for out in output_dirs:
            xml_file = out / "openEMS_model.xml"
            if xml_file.exists():
                print(f"Launching AppCSXCAD viewer for {xml_file}...")
                openems_config.view_in_appcsxcad(xml_file)

if __name__ == '__main__':
    main()
