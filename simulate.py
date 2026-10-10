"""
77 GHz 3-Element Patch Antenna Simulation Script
Simulates the antenna with openEMS FDTD from 76 GHz to 78 GHz.
Saves all outputs, figures, and data into a timestamped subfolder of SimResult/.
"""
import os
import sys
import time
import json
import pathlib
import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure environment is initialized
import openems_config
from antenna_model import PatchAntennaArray77GHz

def calculate_hpbw(theta_deg: np.ndarray, directivity_linear: np.ndarray) -> float:
    """Calculate Half-Power Beamwidth (HPBW) in degrees."""
    # Consider upper hemisphere (-90 to +90 degrees)
    valid_mask = (theta_deg >= -90.0) & (theta_deg <= 90.0)
    theta_sub = theta_deg[valid_mask]
    dir_sub = directivity_linear[valid_mask]
    
    max_idx = np.argmax(dir_sub)
    half_power = dir_sub[max_idx] / 2.0
    
    # Left 3dB point
    left_side = dir_sub[:max_idx]
    if len(left_side) > 0 and np.any(left_side <= half_power):
        left_idx = np.where(left_side <= half_power)[0][-1]
        theta_left = np.interp(half_power, [left_side[left_idx], dir_sub[left_idx+1]], [theta_sub[left_idx], theta_sub[left_idx+1]])
    else:
        theta_left = -90.0
        
    # Right 3dB point
    right_side = dir_sub[max_idx:]
    if len(right_side) > 0 and np.any(right_side <= half_power):
        right_idx = np.where(right_side <= half_power)[0][0] + max_idx
        theta_right = np.interp(half_power, [dir_sub[right_idx], dir_sub[right_idx-1]], [theta_sub[right_idx], theta_sub[right_idx-1]])
    else:
        theta_right = 90.0
        
    return float(np.abs(theta_right - theta_left))

def post_process_directory(output_dir: pathlib.Path,
                           antenna: PatchAntennaArray77GHz = None,
                           sim_duration: float = 0.0,
                           start_dt: datetime.datetime = None,
                           engine: str = 'vulkan') -> None:
    """Perform full post-processing, figure plotting, data export, and report generation."""
    if antenna is None:
        antenna = PatchAntennaArray77GHz()
    if start_dt is None:
        start_dt = datetime.datetime.now()
        
    print("\n" + "=" * 70)
    print("POST-PROCESSING 4-CHANNEL ARRAY SIMULATION RESULTS")
    print(f"Directory: {output_dir}")
    print("=" * 70)
    
    # Re-create ports and nf2ff wrappers
    fdtd, csx, ports, nf2ff = antenna.create_simulation()
    if not isinstance(ports, list):
        ports = [ports]
        
    # Generate antenna visual plots (2D layout, 3D isometric structure, interactive 3D HTML)
    print("Generating antenna visual models and layout diagrams...")
    antenna.draw_layout(str(output_dir / "antenna_layout.png"))
    antenna.draw_3d_view(str(output_dir / "antenna_3d_view.png"))
    antenna.draw_interactive_3d(str(output_dir / "antenna_3d_interactive.html"))
    print("Saved: antenna_layout.png, antenna_3d_view.png, antenna_3d_interactive.html")
    
    # 1. Post-Processing: S-Parameters & Impedance
    print("Analyzing multi-port signals, S-parameters, and mutual coupling...")
    f_band = np.linspace(76.0e9, 78.0e9, 201)  # Target 76 - 78 GHz band
    f_wide = np.linspace(74.0e9, 80.0e9, 301)  # Wideband context
    
    port1 = ports[0]
    port1.CalcPort(str(output_dir), f_wide)
    s11_wide = port1.uf_ref / port1.uf_inc
    s11_wide_db = 20.0 * np.log10(np.abs(s11_wide))
    
    port1.CalcPort(str(output_dir), f_band)
    s11_band = port1.uf_ref / port1.uf_inc
    s11_band_db = 20.0 * np.log10(np.abs(s11_band))
    zin_band = port1.uf_tot / port1.if_tot
    
    # Calculate mutual coupling S_{k1} for all secondary ports
    s_coupled_db = {}
    for ch_idx in range(1, len(ports)):
        p_sec = ports[ch_idx]
        p_sec.CalcPort(str(output_dir), f_band)
        s_k1 = p_sec.uf_tot / port1.uf_inc
        s_k1_db = 20.0 * np.log10(np.maximum(np.abs(s_k1), 1e-6))
        s_coupled_db[f'S{ch_idx+1}1'] = s_k1_db
    
    # Key reflection metrics
    min_s11_idx = np.argmin(s11_band_db)
    min_s11_db = float(s11_band_db[min_s11_idx])
    f_res_ghz = float(f_band[min_s11_idx] / 1e9)
    zin_at_res = zin_band[min_s11_idx]
    
    idx_77 = np.argmin(np.abs(f_band - 77.0e9))
    s11_77_db = float(s11_band_db[idx_77])
    zin_77 = zin_band[idx_77]
    
    # Mutual coupling levels at 77.0 GHz
    coupling_77_db = {k: float(v[idx_77]) for k, v in s_coupled_db.items()}
    
    # -10 dB bandwidth
    match_10db_mask = s11_band_db <= -10.0
    if np.any(match_10db_mask):
        f_match = f_band[match_10db_mask]
        bw_10db_ghz = float((f_match[-1] - f_match[0]) / 1e9)
        bw_10db_pct = float(bw_10db_ghz / 77.0 * 100.0)
    else:
        bw_10db_ghz = 0.0
        bw_10db_pct = 0.0
        
    # 2. Far-Field Radiation Pattern (at 77.0 GHz)
    print("Calculating far-field radiation patterns at 77.0 GHz...")
    theta = np.linspace(-180.0, 180.0, 361)
    phi = [0.0, 90.0]  # phi=0: H-plane (azimuth, xz), phi=90: E-plane (elevation, yz)
    
    nf2ff_res = nf2ff.CalcNF2FF(str(output_dir), 77.0e9, theta, phi, center=[0, 0, 0])
    
    dmax_linear = float(nf2ff_res.Dmax[0])
    dmax_dbi = float(10.0 * np.log10(dmax_linear))
    
    e_norm = nf2ff_res.E_norm[0]
    dir_pattern_h_linear = (e_norm[:, 0] / np.max(e_norm[:, 0]))**2 * dmax_linear
    dir_pattern_e_linear = (e_norm[:, 1] / np.max(e_norm[:, 1]))**2 * dmax_linear
    
    dir_pattern_h_dbi = 10.0 * np.log10(np.maximum(dir_pattern_h_linear, 1e-6))
    dir_pattern_e_dbi = 10.0 * np.log10(np.maximum(dir_pattern_e_linear, 1e-6))
    
    hpbw_h = calculate_hpbw(theta, dir_pattern_h_linear)
    hpbw_e = calculate_hpbw(theta, dir_pattern_e_linear)
    
    idx_theta_0 = np.argmin(np.abs(theta - 0.0))
    idx_theta_180 = np.argmin(np.abs(np.abs(theta) - 180.0))
    fb_ratio_db = float(dir_pattern_e_dbi[idx_theta_0] - dir_pattern_e_dbi[idx_theta_180])
    
    # 3. 4-Channel Synthesized Broadside Beamforming Array Pattern in Azimuth (H-plane)
    k0 = 2 * np.pi / (antenna.lambda0_mm * 1e-3)
    theta_rad = np.deg2rad(theta)
    
    af_complex = np.zeros(len(theta), dtype=complex)
    for x_k_mm in antenna.channel_offsets_x:
        x_k_m = x_k_mm * 1e-3
        psi_k = k0 * x_k_m * np.sin(theta_rad)
        af_complex += np.exp(1j * psi_k)
        
    af_mag_sq = np.abs(af_complex)**2
    array_gain_factor = af_mag_sq / float(antenna.num_channels)
    dir_array_h_linear = dir_pattern_h_linear * array_gain_factor
    dir_array_h_dbi = 10.0 * np.log10(np.maximum(dir_array_h_linear, 1e-6))
    dmax_array_dbi = float(np.max(dir_array_h_dbi))
    hpbw_array_h = calculate_hpbw(theta, dir_array_h_linear)
    
    # 4. Generate and Save Figures
    print("Plotting figures...")
    
    # Figure 1: S11 Return Loss
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), tight_layout=True)
    ax1.plot(f_band / 1e9, s11_band_db, 'b-', linewidth=2.5, label='|S11| (Port 1 Driven)')
    ax1.axhline(-10.0, color='r', linestyle='--', linewidth=1.5, label='-10 dB Specification')
    ax1.axvline(77.0, color='gray', linestyle=':', linewidth=1.2, label='77.0 GHz Center')
    ax1.plot(f_res_ghz, min_s11_db, 'ro', markersize=8, label=f'Min S11: {min_s11_db:.2f} dB @ {f_res_ghz:.2f} GHz')
    ax1.fill_between(f_band / 1e9, -40, -10, where=(s11_band_db <= -10), color='green', alpha=0.15, label='Matched Band (S11 < -10 dB)')
    ax1.set_xlim([76.0, 78.0])
    ax1.set_ylim([max(-35.0, np.floor(min_s11_db / 5) * 5 - 5), 0.0])
    ax1.set_xlabel('Frequency (GHz)', fontsize=11)
    ax1.set_ylabel('|S11| (dB)', fontsize=11)
    ax1.set_title(f'77 GHz {antenna.num_channels}-Channel Array — Port 1 S11 Return Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=9)
    
    ax2.plot(f_wide / 1e9, s11_wide_db, 'k-', linewidth=1.8, label='|S11| (Wideband)')
    ax2.axvspan(76.0, 78.0, color='blue', alpha=0.1, label='Radar Band (76 - 78 GHz)')
    ax2.axhline(-10.0, color='r', linestyle='--', linewidth=1.2)
    ax2.set_xlim([74.0, 80.0])
    ax2.set_ylim([-30.0, 0.0])
    ax2.set_xlabel('Frequency (GHz)', fontsize=11)
    ax2.set_ylabel('|S11| (dB)', fontsize=11)
    ax2.set_title('Wideband Overview (74 - 80 GHz)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='lower left', framealpha=0.9, fontsize=9)
    
    s11_fig_path = output_dir / "s11_reflection.png"
    fig.savefig(s11_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {s11_fig_path.name}")
    
    # Figure 2: Mutual Coupling / Channel Isolation
    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
    colors = ['#d9381e', '#ea580c', '#16a34a', '#2563eb']
    for idx, (param_name, s_db) in enumerate(s_coupled_db.items()):
        dist_mm = (idx + 1) * antenna.channel_spacing
        ax.plot(f_band / 1e9, s_db, linewidth=2.2, color=colors[idx % len(colors)],
                label=f'|{param_name}| (Adjacent $\\Delta x = {dist_mm:.1f}$ mm): {s_db[idx_77]:.1f} dB @ 77 GHz')
    ax.axhline(-25.0, color='gray', linestyle='--', linewidth=1.2, label='-25 dB MIMO Isolation Benchmark')
    ax.axvline(77.0, color='purple', linestyle=':', linewidth=1.2, label='77.0 GHz')
    ax.set_xlim([76.0, 78.0])
    ax.set_ylim([-60.0, -15.0])
    ax.set_xlabel('Frequency (GHz)', fontsize=11)
    ax.set_ylabel('Transmission / Mutual Coupling (dB)', fontsize=11)
    ax.set_title(f'77 GHz {antenna.num_channels}-Channel Array — Inter-Channel Mutual Coupling & Isolation (d = {antenna.channel_spacing:.1f} mm)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    
    coupling_fig_path = output_dir / "mutual_coupling.png"
    fig.savefig(coupling_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {coupling_fig_path.name}")
    
    # Figure 3: Input Impedance
    fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
    ax.plot(f_band / 1e9, np.real(zin_band), 'b-', linewidth=2.0, label='Resistance Re(Zin)')
    ax.plot(f_band / 1e9, np.imag(zin_band), 'r--', linewidth=2.0, label='Reactance Im(Zin)')
    ax.axhline(50.0, color='gray', linestyle=':', linewidth=1.2, label='50 $\\Omega$ Target')
    ax.axhline(0.0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axvline(77.0, color='purple', linestyle=':', linewidth=1.2, label='77.0 GHz')
    ax.set_xlim([76.0, 78.0])
    ax.set_xlabel('Frequency (GHz)', fontsize=11)
    ax.set_ylabel('Impedance ($\\Omega$)', fontsize=11)
    ax.set_title(f'77 GHz Array — Port 1 Input Impedance Zin', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    zin_fig_path = output_dir / "input_impedance.png"
    fig.savefig(zin_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {zin_fig_path.name}")
    
    # Figure 4: Radiation Pattern - Polar Plot (Single Channel + Synthesized Array)
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={'projection': 'polar'}, tight_layout=True)
    theta_rad_plot = np.deg2rad(theta)
    r_min = -25.0
    r_max = 20.0
    
    pat_e_clipped = np.clip(dir_pattern_e_dbi, r_min, r_max)
    pat_h_clipped = np.clip(dir_pattern_h_dbi, r_min, r_max)
    pat_array_clipped = np.clip(dir_array_h_dbi, r_min, r_max)
    
    ax.plot(theta_rad_plot, pat_e_clipped, 'b-', linewidth=2.0, label=f'Single-Ch E-Plane (Elevation, HPBW={hpbw_e:.1f}°)')
    ax.plot(theta_rad_plot, pat_h_clipped, 'orange', linestyle='--', linewidth=1.8, label=f'Single-Ch H-Plane (Azimuth, HPBW={hpbw_h:.1f}°)')
    ax.plot(theta_rad_plot, pat_array_clipped, 'red', linewidth=2.5, label=f'4-Ch Synthesized Broadside Beam (Azimuth, HPBW={hpbw_array_h:.1f}°, Dmax={dmax_array_dbi:.2f} dBi)')
    
    ax.set_theta_zero_location('N')  # 0 deg (broadside +z) at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_rlim(r_min, r_max)
    ax.set_rticks(np.arange(r_min, r_max + 1, 10))
    ax.set_rlabel_position(45)
    ax.set_title(f'77 GHz Far-Field Radiation Pattern (Single-Channel vs 4-Channel Array)\nSynthesized Broadside Directivity = {dmax_array_dbi:.2f} dBi (Single: {dmax_dbi:.2f} dBi)',
                 fontsize=11.5, fontweight='bold', pad=22)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), framealpha=0.9, fontsize=9.0)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    polar_fig_path = output_dir / "radiation_pattern_polar.png"
    fig.savefig(polar_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {polar_fig_path.name}")
    
    # Figure 5: Radiation Pattern - Cartesian Plot
    fig, ax = plt.subplots(figsize=(10, 5.5), tight_layout=True)
    ax.plot(theta, dir_pattern_e_dbi, 'b-', linewidth=2.0, label=f'Single-Ch E-Plane (Elevation, HPBW={hpbw_e:.1f}°)')
    ax.plot(theta, dir_pattern_h_dbi, 'orange', linestyle='--', linewidth=1.8, label=f'Single-Ch H-Plane (Azimuth, HPBW={hpbw_h:.1f}°)')
    ax.plot(theta, dir_array_h_dbi, 'red', linewidth=2.5, label=f'4-Ch Synthesized Array (Azimuth, HPBW={hpbw_array_h:.1f}°, Peak={dmax_array_dbi:.2f} dBi)')
    ax.axhline(dmax_array_dbi, color='green', linestyle=':', linewidth=1.2, label=f'Array Peak: {dmax_array_dbi:.2f} dBi')
    ax.set_xlim([-180.0, 180.0])
    ax.set_ylim([-30.0, 20.0])
    ax.set_xlabel('Angle $\\theta$ (degrees)', fontsize=11)
    ax.set_ylabel('Directivity (dBi)', fontsize=11)
    ax.set_title(f'77 GHz Far-Field Directivity vs Angle — Active Element vs 4-Channel Synthesized Beam', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    
    cart_fig_path = output_dir / "radiation_pattern_cartesian.png"
    fig.savefig(cart_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {cart_fig_path.name}")
    
    # 5. Export CSV Data
    csv_s11_path = output_dir / "s_parameters_data.csv"
    with open(csv_s11_path, 'w', encoding='utf-8') as f:
        f.write("frequency_ghz,s11_db,s21_db,s31_db,s41_db,zin_real_ohm,zin_imag_ohm\n")
        s21_arr = s_coupled_db.get('S21', np.zeros_like(s11_band_db))
        s31_arr = s_coupled_db.get('S31', np.zeros_like(s11_band_db))
        s41_arr = s_coupled_db.get('S41', np.zeros_like(s11_band_db))
        for idx_f in range(len(f_band)):
            f_val = f_band[idx_f] / 1e9
            f.write(f"{f_val:.4f},{s11_band_db[idx_f]:.4f},{s21_arr[idx_f]:.4f},{s31_arr[idx_f]:.4f},{s41_arr[idx_f]:.4f},{zin_band[idx_f].real:.4f},{zin_band[idx_f].imag:.4f}\n")
    print(f"Exported CSV data: {csv_s11_path.name}")
    
    csv_pattern_path = output_dir / "radiation_pattern_77ghz.csv"
    with open(csv_pattern_path, 'w', encoding='utf-8') as f:
        f.write("theta_deg,single_ch_h_plane_dbi,single_ch_e_plane_dbi,array_4ch_h_plane_dbi\n")
        for th_val, dh_val, de_val, da_val in zip(theta, dir_pattern_h_dbi, dir_pattern_e_dbi, dir_array_h_dbi):
            f.write(f"{th_val:.1f},{dh_val:.4f},{de_val:.4f},{da_val:.4f}\n")
    print(f"Exported CSV data: {csv_pattern_path.name}")
    
    # 6. Summary and Reports
    end_dt = datetime.datetime.now()
    summary = {
        "simulation_info": {
            "antenna_type": f"{antenna.num_channels}-Channel 3-Element Series-Fed Patch Array",
            "num_channels": antenna.num_channels,
            "channel_spacing_mm": antenna.channel_spacing,
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "total_elapsed_seconds": round(sim_duration, 2),
            "output_directory": str(output_dir),
            "engine": engine
        },
        "frequency_specifications": {
            "center_frequency_ghz": 77.0,
            "band_start_ghz": 76.0,
            "band_stop_ghz": 78.0,
            "resonant_frequency_ghz": round(f_res_ghz, 3),
            "min_s11_db": round(min_s11_db, 2),
            "s11_at_77ghz_db": round(s11_77_db, 2),
            "bandwidth_10db_ghz": round(bw_10db_ghz, 3),
            "bandwidth_10db_percent": round(bw_10db_pct, 2)
        },
        "mutual_coupling_at_77ghz": {
            k: round(v, 2) for k, v in coupling_77_db.items()
        },
        "impedance_results": {
            "zin_at_resonance": f"{zin_at_res.real:.2f} + {zin_at_res.imag:.2f}j Ohm",
            "zin_at_77ghz": f"{zin_77.real:.2f} + {zin_77.imag:.2f}j Ohm",
            "system_impedance_ohm": 50.0
        },
        "single_channel_radiation_at_77ghz": {
            "peak_directivity_dbi": round(dmax_dbi, 2),
            "peak_directivity_linear": round(dmax_linear, 2),
            "hpbw_e_plane_deg": round(hpbw_e, 1),
            "hpbw_h_plane_deg": round(hpbw_h, 1),
            "front_to_back_ratio_db": round(fb_ratio_db, 2)
        },
        "synthesized_4channel_array_at_77ghz": {
            "peak_directivity_dbi": round(dmax_array_dbi, 2),
            "azimuth_hpbw_deg": round(hpbw_array_h, 1),
            "array_gain_db": round(dmax_array_dbi - dmax_dbi, 2)
        },
        "geometry_parameters_mm": {
            "num_channels": antenna.num_channels,
            "channel_spacing_mm": antenna.channel_spacing,
            "channel_offsets_x_mm": antenna.channel_offsets_x,
            "elements_per_channel": antenna.num_elements,
            "patch_length": antenna.patch_length,
            "patch_width": antenna.patch_width,
            "feed_width_50ohm": antenna.feed_width,
            "transformer_width": antenna.trans_width,
            "transformer_length": antenna.trans_length,
            "connecting_line_width": antenna.connect_width,
            "connecting_line_length": antenna.connect_length,
            "substrate_width_mm": antenna.sub_w,
            "substrate_length_mm": antenna.sub_l,
            "stackup_type": "4-Layer Automotive mmWave Radar" if antenna.is_4layer else "Isolated RF Cavity",
            "dielectric_1_material": "Rogers RO3003",
            "dielectric_1_thickness_mm": antenna.sub_h,
            "layer_2_ground": "Continuous Copper RF Ground Plane (z = 0)",
            "dielectric_2_material": "FR-4 Structural Core" if antenna.is_4layer else "None",
            "dielectric_2_thickness_mm": antenna.h_core if antenna.is_4layer else 0.0,
            "layer_4_ground": "Continuous Bottom Ground / Shielding Plane" if antenna.is_4layer else "None",
            "total_board_thickness_mm": round(antenna.sub_h + (antenna.h_core if antenna.is_4layer else 0.0), 3),
            "cells_per_wavelength": antenna.cells_per_wavelength,
            "max_mesh_resolution_um": round(antenna.lambda0_mm / antenna.cells_per_wavelength * 1e3, 1)
        }
    }
    
    summary_json_path = output_dir / "simulation_summary.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    coupling_report_str = "\n".join([f"  - {k} ({ (i+1)*antenna.channel_spacing:.1f} mm spacing): {v:.2f} dB"
                                     for i, (k, v) in enumerate(coupling_77_db.items())])
    
    report_txt_path = output_dir / "simulation_report.txt"
    report_content = f"""================================================================================
77 GHz {antenna.num_channels}-CHANNEL PATCH ANTENNA ARRAY SIMULATION REPORT (4-LAYER STACKUP)
===============================================================================
Timestamp Started:       {start_dt.strftime("%Y-%m-%d %H:%M:%S")}
Timestamp Finished:      {end_dt.strftime("%Y-%m-%d %H:%M:%S")}
Total Execution Time:    {sim_duration:.2f} seconds
Output Directory:        {output_dir}

================================================================================
ARRAY ARCHITECTURE & CONFIGURATION
================================================================================
Number of Channels:           {antenna.num_channels} parallel series-fed patch arrays
Channel Spacing (center-to-center): {antenna.channel_spacing:.2f} mm (~{antenna.channel_spacing/antenna.lambda0_mm:.2f} lambda_0)
Channel Offsets (X-axis):     {antenna.channel_offsets_x} mm
Port Feeding Architecture:    4 Independent 50-Ohm Ports (Port 1 Driven, Ports 2-4 Terminated)
Stackup Architecture:         {"4-Layer Automotive mmWave Radar" if antenna.is_4layer else "Isolated Cavity"}
Substrate Dimensions (W x L): {antenna.sub_w:.2f} x {antenna.sub_l:.2f} mm
Total Board Thickness:        ~{(antenna.sub_h + (antenna.h_core if antenna.is_4layer else 0.0)):.2f} mm

================================================================================
RF PERFORMANCE SUMMARY (76.0 GHz to 78.0 GHz)
================================================================================
Resonant Frequency (f_res):       {f_res_ghz:.3f} GHz
Min Return Loss (S11 @ f_res):    {min_s11_db:.2f} dB
Return Loss @ 77.0 GHz:           {s11_77_db:.2f} dB
Input Impedance @ 77.0 GHz:       {zin_77.real:.2f} + {zin_77.imag:.2f}j Ohm
-10 dB Bandwidth:                 {bw_10db_ghz:.3f} GHz ({bw_10db_pct:.2f}%)

INTER-CHANNEL MUTUAL COUPLING / ISOLATION AT 77.0 GHz:
{coupling_report_str}

FAR-FIELD RADIATION AT 77.0 GHz:
Single-Channel Active Element:
  - Peak Directivity (D_max):     {dmax_dbi:.2f} dBi ({dmax_linear:.2f} linear)
  - E-Plane HPBW (yz, Elevation): {hpbw_e:.1f} degrees
  - H-Plane HPBW (xz, Azimuth):   {hpbw_h:.1f} degrees
  - Front-to-Back Ratio:          {fb_ratio_db:.2f} dB

4-Channel Synthesized Broadside Beamforming Array:
  - Peak Directivity (D_array):   {dmax_array_dbi:.2f} dBi (+{dmax_array_dbi - dmax_dbi:.2f} dB array gain)
  - Azimuth HPBW (xz, Azimuth):   {hpbw_array_h:.1f} degrees (focused down from {hpbw_h:.1f} deg)

================================================================================
GENERATED OUTPUT FILES
================================================================================
1. antenna_layout.png             - 2D layout geometry showing all 4 channels and 4-layer stackup
2. antenna_3d_view.png             - 3D isometric perspective structural plot
3. antenna_3d_interactive.html     - Interactive 3D WebGL model (orbit/zoom/pan in browser)
4. s11_reflection.png             - S11 reflection coefficient vs frequency (76-78 GHz)
5. mutual_coupling.png            - Inter-channel mutual coupling / isolation (S21, S31, S41)
6. input_impedance.png            - Complex input impedance Zin vs frequency
7. radiation_pattern_polar.png    - 2D polar directivity patterns (single ch vs 4-ch array)
8. radiation_pattern_cartesian.png- Cartesian directivity vs elevation angle
9. s_parameters_data.csv          - Numerical S-parameter matrix (S11, S21, S31, S41, Zin)
10. radiation_pattern_77ghz.csv   - Numerical far-field directivity table
11. openEMS_model.xml             - CSXCAD full 3D simulation model
12. simulation_summary.json       - Structured JSON results
13. simulation_report.txt         - This summary report
================================================================================
"""
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("\n" + "=" * 70)
    print("4-CHANNEL ARRAY SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Resonant Frequency:       {f_res_ghz:.3f} GHz")
    print(f"Port 1 Return Loss S11:   {s11_77_db:.2f} dB")
    for k, v in coupling_77_db.items():
        print(f"Coupling {k} @ 77 GHz:      {v:.2f} dB")
    print(f"Single-Channel Dmax:      {dmax_dbi:.2f} dBi")
    print(f"4-Ch Broadside Dmax:      {dmax_array_dbi:.2f} dBi (HPBW: {hpbw_array_h:.1f} deg)")
    print(f"Results saved in:         {output_dir}")
    print("=" * 70)

def run_simulation(engine: str = 'vulkan', num_threads: int = 0, cells_per_wavelength: float = 40.0) -> pathlib.Path:
    # 1. Record start time and create timestamped directory
    start_dt = datetime.datetime.now()
    timestamp_str = start_dt.strftime("%Y-%m-%d_%H-%M-%S")
    
    project_root = pathlib.Path(__file__).resolve().parent
    sim_result_root = project_root / "SimResult"
    output_dir = sim_result_root / timestamp_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("77 GHz 4-CHANNEL PATCH ANTENNA ARRAY SIMULATION (5.0 mm SPACING)")
    print("=" * 70)
    print(f"Simulation Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory:     {output_dir}")
    print(f"Channels:             4 channels side-by-side, 5.0 mm center-to-center")
    print(f"Stackup Architecture: 4-Layer Automotive mmWave Radar")
    print(f"Mesh Resolution:      {cells_per_wavelength:.1f} cells/lambda_0 (~{(3.896/cells_per_wavelength)*1e3:.1f} um cell)")
    print(f"Band of Interest:     76.0 GHz to 78.0 GHz (Center: 77.0 GHz)")
    print(f"Solver Engine:        {engine}")
    print("=" * 70)
    
    # 2. Instantiate 4-Channel Antenna Model
    antenna = PatchAntennaArray77GHz(
        f_center=77.0e9,
        f_start=76.0e9,
        f_stop=78.0e9,
        num_channels=4,
        channel_spacing=5.000,
        num_elements=3,
        patch_length=1.070,   # mm (tuned for 77.0 GHz)
        patch_width=1.350,    # mm
        feed_width=0.320,     # mm (50 Ohm line)
        feed_length=1.000,    # mm
        trans_width=0.100,    # mm (quarter-wave transformer)
        trans_length=0.615,   # mm
        connect_width=0.150,  # mm
        connect_length=1.340, # mm
        is_4layer=True,
        cells_per_wavelength=cells_per_wavelength
    )
    
    # Generate antenna visual models and layout diagrams
    print("\nGenerating antenna layout diagrams and 3D visual models...")
    antenna.draw_layout(str(output_dir / "antenna_layout.png"))
    antenna.draw_3d_view(str(output_dir / "antenna_3d_view.png"))
    antenna.draw_interactive_3d(str(output_dir / "antenna_3d_interactive.html"))
    print("Saved: antenna_layout.png, antenna_3d_view.png, antenna_3d_interactive.html")
    
    # 3. Create openEMS simulation & mesh
    print("\nBuilding geometry and FDTD mesh...")
    fdtd, csx, ports, nf2ff = antenna.create_simulation(end_criteria=1e-4, nr_ts=20000)
    
    # Export CSX model XML
    csx_xml_path = output_dir / "openEMS_model.xml"
    csx.Write2XML(str(csx_xml_path))
    print(f"Exported CSX model XML: {csx_xml_path.name}")
    
    # 4. Execute FDTD Simulation
    print("\nStarting openEMS FDTD solver...")
    sim_t0 = time.time()
    run_kwargs = {'cleanup': True}
    if engine:
        run_kwargs['engine'] = engine
    if num_threads > 0:
        run_kwargs['numThreads'] = num_threads
        
    fdtd.Run(str(output_dir), **run_kwargs)
    sim_duration = time.time() - sim_t0
    print(f"FDTD simulation completed in {sim_duration:.2f} seconds.")
    
    # 5. Execute Post-Processing
    post_process_directory(
        output_dir=output_dir,
        antenna=antenna,
        sim_duration=sim_duration,
        start_dt=start_dt,
        engine=engine
    )
    
    return output_dir

if __name__ == '__main__':
    run_simulation()
