"""
Multi-Antenna openEMS FDTD Simulation and Post-Processing Engine.
Supports proven Texas Instruments reference antennas:
- TI 77 GHz RX series-fed patch array (SWRU553A)
- TI 2.4 GHz MIFA PCB antenna (AN043 / SWRA117D)
- TI 2.4 GHz Inverted-F antenna (SWRU120D / DN0007)
"""
import os
import sys
import time
import json
import pathlib
import datetime
import subprocess
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure environment is initialized
import openems_config
import antenna_model
from antenna_model import BaseAntennaModel

class TeeLogger:
    """Duplicate stdout streams to both the terminal and a log file."""
    def __init__(self, log_path: pathlib.Path):
        self.log_path = log_path
        self.terminal = sys.stdout
        self.log_file = None
        self._active = False

    def start(self):
        if not self._active:
            self.log_file = open(self.log_path, 'a', encoding='utf-8')
            sys.stdout = self
            self._active = True

    def stop(self):
        if self._active:
            sys.stdout = self.terminal
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            self._active = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

def run_openems_solver(output_dir: pathlib.Path,
                       xml_path: pathlib.Path,
                       engine: str = 'vulkan',
                       num_threads: int = 0,
                       log_path: pathlib.Path = None,
                       vulkan_profile: bool = False,
                       vulkan_pml: str = 'fused') -> int:
    """Execute openEMS solver executable while streaming and logging all output."""
    openems_exe = openems_config.get_openems_path() / 'openEMS.exe'
    
    if not openems_exe.exists():
        return -1
        
    cmd = [str(openems_exe), str(xml_path)]
    if engine:
        cmd.append(f'--engine={engine}')
    if engine in ('vulkan', 'gpu'):
        cmd.append('--vulkan-fd=gpu')
        cmd.append(f'--vulkan-pml={vulkan_pml}')
        if vulkan_profile:
            cmd.append('--vulkan-profile')
    if num_threads > 0:
        cmd.append(f'--numThreads={num_threads}')
    cmd.append('--dump-statistics')
    
    if log_path is None:
        log_path = output_dir / "openEMS.log"
        
    with open(log_path, 'w', encoding='utf-8') as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(output_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        proc.wait()
        
    return proc.returncode

def calculate_hpbw(theta_deg: np.ndarray, directivity_linear: np.ndarray) -> float:
    """Calculate Half-Power Beamwidth (HPBW) in degrees."""
    valid_mask = (theta_deg >= -90.0) & (theta_deg <= 90.0)
    theta_sub = theta_deg[valid_mask]
    dir_sub = directivity_linear[valid_mask]
    
    if len(dir_sub) == 0 or np.max(dir_sub) <= 0:
        return 0.0
        
    max_idx = np.argmax(dir_sub)
    half_power = dir_sub[max_idx] / 2.0
    
    left_side = dir_sub[:max_idx]
    if len(left_side) > 0 and np.any(left_side <= half_power):
        left_idx = np.where(left_side <= half_power)[0][-1]
        theta_left = np.interp(half_power, [left_side[left_idx], dir_sub[left_idx+1]], [theta_sub[left_idx], theta_sub[left_idx+1]])
    else:
        theta_left = -90.0
        
    right_side = dir_sub[max_idx:]
    if len(right_side) > 0 and np.any(right_side <= half_power):
        right_idx = np.where(right_side <= half_power)[0][0] + max_idx
        theta_right = np.interp(half_power, [dir_sub[right_idx], dir_sub[right_idx-1]], [theta_sub[right_idx], theta_sub[right_idx-1]])
    else:
        theta_right = 90.0
        
    return float(np.abs(theta_right - theta_left))

def post_process_directory(output_dir: pathlib.Path,
                           antenna: BaseAntennaModel = None,
                           sim_duration: float = 0.0,
                           start_dt: datetime.datetime = None,
                           engine: str = 'vulkan',
                           channel: str = '1',
                           max_timesteps: int = 40000) -> None:
    """Perform full post-processing, figure plotting, data export, and report generation."""
    if antenna is None:
        antenna = antenna_model.get_antenna_model('ti_77ghz_rx')
    if start_dt is None:
        start_dt = datetime.datetime.now()
        
    sim_log_path = output_dir / "simulation.log"
    logger = None
    if not isinstance(sys.stdout, TeeLogger):
        logger = TeeLogger(sim_log_path)
        logger.start()

    is_all_channels = str(channel).strip().lower() in ('all', 'array') and antenna.num_channels > 1
    if not is_all_channels:
        try:
            ch_active = int(channel)
        except Exception:
            ch_active = 1
        if ch_active < 1 or ch_active > antenna.num_channels:
            ch_active = 1
        mode_str = f"Channel {ch_active} Excited" if antenna.num_channels > 1 else "Single 50-Ohm Port Driven"
    else:
        ch_active = 1
        mode_str = f"All {antenna.num_channels} Channels Driven Simultaneously (Active Broadside Array)"

    print("\n" + "=" * 75)
    print(f"POST-PROCESSING: {antenna.title.upper()}")
    print(f"Directory:           {output_dir}")
    print(f"Reference Doc:       {antenna.doc_reference}")
    print(f"Excitation Mode:     {mode_str}")
    print(f"Max Timesteps:       {max_timesteps}")
    print("=" * 75)
    
    # Re-create ports and nf2ff wrappers
    fdtd, csx, ports, nf2ff = antenna.create_simulation(end_criteria=1e-4, nr_ts=max_timesteps, active_channels=channel)
    if not isinstance(ports, list):
        ports = [ports]
        
    # Generate presentation files
    print("\nGenerating presentation visual diagrams and interactive 3D HTML model...")
    antenna.draw_layout(str(output_dir / "antenna_layout.png"))
    antenna.draw_3d_view(str(output_dir / "antenna_3d_view.png"))
    antenna.draw_interactive_3d(str(output_dir / "antenna_3d_interactive.html"))
    print("Saved: antenna_layout.png, antenna_3d_view.png, antenna_3d_interactive.html")
    
    # 1. Post-Processing: S-Parameters & Impedance
    print(f"\nAnalyzing port signals and S-parameters ({mode_str})...")
    f_band = np.linspace(antenna.f_start, antenna.f_stop, 201)
    f_wide = np.linspace(antenna.f_wide_start, antenna.f_wide_stop, 301)
    
    s_coupled_db = {}
    s_active_all_band_db = {}
    s_active_all_wide_db = {}
    zin_all = {}
    
    if not is_all_channels:
        port_active = ports[ch_active - 1]
        port_active.CalcPort(str(output_dir), f_wide)
        s11_wide = port_active.uf_ref / port_active.uf_inc
        s11_wide_db = 20.0 * np.log10(np.abs(s11_wide))
        
        port_active.CalcPort(str(output_dir), f_band)
        s11_band = port_active.uf_ref / port_active.uf_inc
        s11_band_db = 20.0 * np.log10(np.abs(s11_band))
        zin_band = port_active.uf_tot / port_active.if_tot
        
        if len(ports) > 1:
            for ch_idx in range(len(ports)):
                if ch_idx == (ch_active - 1):
                    continue
                p_sec = ports[ch_idx]
                p_sec.CalcPort(str(output_dir), f_band)
                s_kj = p_sec.uf_tot / port_active.uf_inc
                s_kj_db = 20.0 * np.log10(np.maximum(np.abs(s_kj), 1e-6))
                s_coupled_db[f'S{ch_idx+1}{ch_active}'] = s_kj_db
    else:
        for ch_idx in range(len(ports)):
            p = ports[ch_idx]
            p.CalcPort(str(output_dir), f_wide)
            s_w = p.uf_ref / p.uf_inc
            s_active_all_wide_db[f'Port {ch_idx+1}'] = 20.0 * np.log10(np.abs(s_w))
            
            p.CalcPort(str(output_dir), f_band)
            s_b = p.uf_ref / p.uf_inc
            s_active_all_band_db[f'Port {ch_idx+1}'] = 20.0 * np.log10(np.abs(s_b))
            zin_all[f'Port {ch_idx+1}'] = p.uf_tot / p.if_tot
            
        port_active = ports[0]
        s11_wide_db = s_active_all_wide_db['Port 1']
        s11_band_db = s_active_all_band_db['Port 1']
        zin_band = zin_all['Port 1']
        s_coupled_db = s_active_all_band_db

    # Key reflection metrics
    min_s11_idx = np.argmin(s11_band_db)
    min_s11_db = float(s11_band_db[min_s11_idx])
    f_res_ghz = float(f_band[min_s11_idx] / 1e9)
    zin_at_res = zin_band[min_s11_idx]
    
    idx_center = np.argmin(np.abs(f_band - antenna.f_center))
    s11_center_db = float(s11_band_db[idx_center])
    zin_center = zin_band[idx_center]
    
    # -10 dB bandwidth
    match_10db_mask = s11_band_db <= -10.0
    if np.any(match_10db_mask):
        f_match = f_band[match_10db_mask]
        bw_10db_ghz = float((f_match[-1] - f_match[0]) / 1e9)
        bw_10db_pct = float(bw_10db_ghz / (antenna.f_center / 1e9) * 100.0)
    else:
        bw_10db_ghz = 0.0
        bw_10db_pct = 0.0

    # 2. Far-Field Radiation Pattern (at antenna.f_center)
    f_center_ghz = antenna.f_center / 1e9
    print(f"Calculating far-field radiation patterns at {f_center_ghz:.2f} GHz...")
    theta = np.linspace(-180.0, 180.0, 361)
    phi = [0.0, 90.0]  # phi=0: H-plane (xz), phi=90: E-plane (yz)
    
    nf2ff_res = nf2ff.CalcNF2FF(str(output_dir), antenna.f_center, theta, phi, center=[0, 0, 0])
    dmax_linear = float(nf2ff_res.Dmax[0])
    dmax_dbi = float(10.0 * np.log10(max(dmax_linear, 1e-4)))
    
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

    if antenna.num_channels > 1 and not is_all_channels:
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
    else:
        dmax_array_dbi = dmax_dbi
        hpbw_array_h = hpbw_h
        dir_array_h_dbi = dir_pattern_h_dbi

    # 3. Generate and Save Figures
    print("Generating simulation response figures...")
    
    # Figure 1: S11 Return Loss
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), tight_layout=True)
    f_band_ghz = f_band / 1e9
    f_wide_ghz = f_wide / 1e9
    
    if not is_all_channels:
        ax1.plot(f_band_ghz, s11_band_db, 'b-', linewidth=2.5, label=f'|S11| Return Loss')
        ax1.axhline(-10.0, color='r', linestyle='--', linewidth=1.5, label='-10 dB Specification')
        ax1.axvline(f_center_ghz, color='purple', linestyle=':', linewidth=1.2, label=f'{f_center_ghz:.2f} GHz Center')
        ax1.plot(f_res_ghz, min_s11_db, 'ro', markersize=8, label=f'Resonance: {min_s11_db:.2f} dB @ {f_res_ghz:.3f} GHz')
        ax1.fill_between(f_band_ghz, -40, -10, where=(s11_band_db <= -10), color='green', alpha=0.15, label='Matched Band (S11 <= -10 dB)')
    else:
        colors_all = ['#2563eb', '#16a34a', '#ea580c', '#d9381e']
        for idx_ch, (p_name, p_sdb) in enumerate(s_active_all_band_db.items()):
            ax1.plot(f_band_ghz, p_sdb, linewidth=2.2, color=colors_all[idx_ch % len(colors_all)], label=f'{p_name} Active S11')
        ax1.axhline(-10.0, color='r', linestyle='--', linewidth=1.5, label='-10 dB Spec')
        ax1.axvline(f_center_ghz, color='purple', linestyle=':', linewidth=1.2)
        
    ax1.set_xlim([f_band_ghz[0], f_band_ghz[-1]])
    ax1.set_ylim([min(-35.0, np.floor(min_s11_db / 5) * 5 - 5), 0.0])
    ax1.set_xlabel('Frequency (GHz)', fontsize=11)
    ax1.set_ylabel('|S-Parameter| (dB)', fontsize=11)
    ax1.set_title(f'{antenna.title} — Return Loss / Reflection (Operating Band)', fontsize=11.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=9)
    
    # Wideband panel
    ax2.plot(f_wide_ghz, s11_wide_db, 'k-', linewidth=1.8, label='Wideband S11')
    ax2.axvspan(f_band_ghz[0], f_band_ghz[-1], color='blue', alpha=0.1, label=f'Operating Band ({f_band_ghz[0]:.2f} - {f_band_ghz[-1]:.2f} GHz)')
    ax2.axhline(-10.0, color='r', linestyle='--', linewidth=1.2)
    ax2.set_xlim([f_wide_ghz[0], f_wide_ghz[-1]])
    ax2.set_ylim([-30.0, 0.0])
    ax2.set_xlabel('Frequency (GHz)', fontsize=11)
    ax2.set_ylabel('|S-Parameter| (dB)', fontsize=11)
    ax2.set_title('Wideband Overview', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='lower left', framealpha=0.9, fontsize=9)
    
    s11_fig_path = output_dir / "s11_reflection.png"
    fig.savefig(s11_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {s11_fig_path.name}")
    
    # Figure 2: Input Impedance
    fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
    ax.plot(f_band_ghz, np.real(zin_band), 'b-', linewidth=2.0, label='Resistance Re(Zin)')
    ax.plot(f_band_ghz, np.imag(zin_band), 'r--', linewidth=2.0, label='Reactance Im(Zin)')
    ax.axhline(50.0, color='gray', linestyle=':', linewidth=1.2, label='50 Target')
    ax.axhline(0.0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axvline(f_center_ghz, color='purple', linestyle=':', linewidth=1.2, label=f'{f_center_ghz:.2f} GHz')
    ax.set_xlim([f_band_ghz[0], f_band_ghz[-1]])
    ax.set_xlabel('Frequency (GHz)', fontsize=11)
    ax.set_ylabel('Impedance (Ohm)', fontsize=11)
    ax.set_title(f'{antenna.title} — Complex Input Impedance Zin', fontsize=11.5, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
    
    zin_fig_path = output_dir / "input_impedance.png"
    fig.savefig(zin_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {zin_fig_path.name}")
    
    # Figure 3: Mutual Coupling (if multi-channel)
    if len(s_coupled_db) > 0:
        fig, ax = plt.subplots(figsize=(9, 5.5), tight_layout=True)
        colors = ['#d9381e', '#ea580c', '#16a34a', '#2563eb']
        for idx, (param_name, s_db) in enumerate(s_coupled_db.items()):
            ax.plot(f_band_ghz, s_db, linewidth=2.2, color=colors[idx % len(colors)], label=f'|{param_name}|')
        ax.axhline(-25.0, color='gray', linestyle='--', linewidth=1.2, label='-25 dB Isolation Benchmark')
        ax.axvline(f_center_ghz, color='purple', linestyle=':', linewidth=1.2)
        ax.set_xlim([f_band_ghz[0], f_band_ghz[-1]])
        ax.set_xlabel('Frequency (GHz)', fontsize=11)
        ax.set_ylabel('Coupling (dB)', fontsize=11)
        ax.set_title(f'{antenna.title} — Inter-Channel Coupling / Isolation', fontsize=11.5, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
        coupling_fig_path = output_dir / "mutual_coupling.png"
        fig.savefig(coupling_fig_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {coupling_fig_path.name}")

    # Figure 4: Radiation Pattern - Polar
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={'projection': 'polar'}, tight_layout=True)
    theta_rad_plot = np.deg2rad(theta)
    r_min = -25.0
    r_max = max(15.0, np.ceil(dmax_array_dbi / 5) * 5 + 5)
    
    pat_e_clipped = np.clip(dir_pattern_e_dbi, r_min, r_max)
    pat_h_clipped = np.clip(dir_pattern_h_dbi, r_min, r_max)
    
    ax.plot(theta_rad_plot, pat_e_clipped, 'b-', linewidth=2.0, label=f'E-Plane (Elevation, HPBW={hpbw_e:.1f}°)')
    ax.plot(theta_rad_plot, pat_h_clipped, 'orange', linestyle='--', linewidth=1.8, label=f'H-Plane (Azimuth, HPBW={hpbw_h:.1f}°)')
    if antenna.num_channels > 1:
        pat_arr_clipped = np.clip(dir_array_h_dbi, r_min, r_max)
        ax.plot(theta_rad_plot, pat_arr_clipped, 'red', linewidth=2.5, label=f'Array Broadside (HPBW={hpbw_array_h:.1f}°, Peak={dmax_array_dbi:.2f} dBi)')
        
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(r_min, r_max)
    ax.set_title(f'{antenna.title} — Radiation Pattern @ {f_center_ghz:.2f} GHz\nPeak Directivity: {dmax_dbi:.2f} dBi',
                 fontsize=11.5, fontweight='bold', pad=22)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), framealpha=0.9, fontsize=9.0)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    polar_fig_path = output_dir / "radiation_pattern_polar.png"
    fig.savefig(polar_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {polar_fig_path.name}")
    
    # Figure 5: Radiation Pattern - Cartesian
    fig, ax = plt.subplots(figsize=(10, 5.5), tight_layout=True)
    ax.plot(theta, dir_pattern_e_dbi, 'b-', linewidth=2.0, label=f'E-Plane (Elevation, HPBW={hpbw_e:.1f}°)')
    ax.plot(theta, dir_pattern_h_dbi, 'orange', linestyle='--', linewidth=1.8, label=f'H-Plane (Azimuth, HPBW={hpbw_h:.1f}°)')
    if antenna.num_channels > 1:
        ax.plot(theta, dir_array_h_dbi, 'red', linewidth=2.5, label=f'Array Broadside (HPBW={hpbw_array_h:.1f}°, Peak={dmax_array_dbi:.2f} dBi)')
    ax.axhline(dmax_dbi, color='green', linestyle=':', linewidth=1.2, label=f'Peak Directivity: {dmax_dbi:.2f} dBi')
    ax.set_xlim([-180.0, 180.0])
    ax.set_ylim([-30.0, r_max])
    ax.set_xlabel('Angle Theta (degrees)', fontsize=11)
    ax.set_ylabel('Directivity (dBi)', fontsize=11)
    ax.set_title(f'{antenna.title} — Directivity vs Angle @ {f_center_ghz:.2f} GHz', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    
    cart_fig_path = output_dir / "radiation_pattern_cartesian.png"
    fig.savefig(cart_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {cart_fig_path.name}")
    
    # 4. Export CSV Data
    csv_s11_path = output_dir / "s_parameters_data.csv"
    with open(csv_s11_path, 'w', encoding='utf-8') as f:
        headers = ["frequency_ghz", "s11_db", "zin_real_ohm", "zin_imag_ohm"]
        f.write(",".join(headers) + "\n")
        for idx_f in range(len(f_band)):
            f_val = f_band[idx_f] / 1e9
            f.write(f"{f_val:.4f},{s11_band_db[idx_f]:.4f},{zin_band[idx_f].real:.4f},{zin_band[idx_f].imag:.4f}\n")
    print(f"Exported CSV data: {csv_s11_path.name}")
    
    csv_pattern_path = output_dir / "radiation_pattern_data.csv"
    with open(csv_pattern_path, 'w', encoding='utf-8') as f:
        f.write("theta_deg,h_plane_dbi,e_plane_dbi,array_broadside_dbi\n")
        for th_val, dh_val, de_val, da_val in zip(theta, dir_pattern_h_dbi, dir_pattern_e_dbi, dir_array_h_dbi):
            f.write(f"{th_val:.1f},{dh_val:.4f},{de_val:.4f},{da_val:.4f}\n")
    print(f"Exported CSV data: {csv_pattern_path.name}")
    
    # 5. Machine-Readable Summary JSON
    end_dt = datetime.datetime.now()
    proven_specs = antenna.get_proven_specs()
    summary = {
        "antenna_model": {
            "name": antenna.name,
            "title": antenna.title,
            "reference_document": antenna.doc_reference,
            "num_channels": antenna.num_channels,
            "operating_band": f"{antenna.f_start/1e9:.3f} - {antenna.f_stop/1e9:.3f} GHz",
            "center_frequency_ghz": round(antenna.f_center / 1e9, 3)
        },
        "simulation_info": {
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(sim_duration, 2),
            "output_directory": str(output_dir),
            "engine": engine,
            "max_timesteps": max_timesteps
        },
        "simulated_rf_metrics": {
            "resonant_frequency_ghz": round(f_res_ghz, 3),
            "min_s11_db": round(min_s11_db, 2),
            "s11_at_nominal_center_db": round(s11_center_db, 2),
            "bandwidth_10db_ghz": round(bw_10db_ghz, 3),
            "bandwidth_10db_percent": round(bw_10db_pct, 2),
            "zin_at_resonance_ohm": f"{zin_at_res.real:.2f} + {zin_at_res.imag:.2f}j",
            "zin_at_nominal_center_ohm": f"{zin_center.real:.2f} + {zin_center.imag:.2f}j",
            "peak_directivity_dbi": round(dmax_dbi, 2),
            "hpbw_e_plane_deg": round(hpbw_e, 1),
            "hpbw_h_plane_deg": round(hpbw_h, 1),
            "front_to_back_ratio_db": round(fb_ratio_db, 2)
        },
        "proven_benchmark_specifications": proven_specs
    }
    
    summary_path = output_dir / "simulation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"Exported Summary JSON: {summary_path.name}")
    
    # 6. Formatted Human-Readable Engineering Report
    report_path = output_dir / "simulation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"OPENEMS FDTD ELECTROMAGNETIC SIMULATION REPORT\n")
        f.write(f"Antenna: {antenna.title}\n")
        f.write(f"Reference Document: {antenna.doc_reference}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. EXECUTIVE SUMMARY & BENCHMARK COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Performance Metric':<30} | {'Simulated openEMS':<22} | {'TI Published Benchmark':<24}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Center Frequency':<30} | {f_center_ghz:.3f} GHz{'':<13} | {antenna.get_proven_specs().get('operating_band', 'N/A'):<24}\n")
        f.write(f"{'Resonant Frequency':<30} | {f_res_ghz:.3f} GHz{'':<13} | Tuned to ISM/Radar band\n")
        f.write(f"{'Return Loss (S11 min)':<30} | {min_s11_db:.2f} dB{'':<14} | {antenna.get_proven_specs().get('proven_s11_reflection_db', '< -10 dB'):<24}\n")
        f.write(f"{'Input Impedance (Zin)':<30} | {zin_center.real:.1f} + {zin_center.imag:.1f}j Ohm{'':<6} | 50.0 Ohm target\n")
        f.write(f"{'-10 dB Bandwidth':<30} | {bw_10db_ghz*1e3:.1f} MHz ({bw_10db_pct:.1f}%){'':<4} | {antenna.get_proven_specs().get('proven_10db_bandwidth_mhz', '> 100 MHz'):<24}\n")
        f.write(f"{'Peak Directivity / Gain':<30} | {dmax_dbi:.2f} dBi{'':<13} | {antenna.get_proven_specs().get('proven_single_element_gain_dbi', antenna.get_proven_specs().get('proven_gain_xz_plane_dbi', 'N/A')):<24}\n")
        f.write(f"{'E-Plane (Elevation) HPBW':<30} | {hpbw_e:.1f} deg{'':<14} | {antenna.get_proven_specs().get('proven_elevation_hpbw_deg', 'N/A'):<24}\n")
        f.write(f"{'H-Plane (Azimuth) HPBW':<30} | {hpbw_h:.1f} deg{'':<14} | {antenna.get_proven_specs().get('proven_azimuth_hpbw_deg', 'Omnidirectional'):<24}\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("2. PROVEN DESIGN SPECIFICATIONS\n")
        f.write("-" * 80 + "\n")
        for k, v in proven_specs.items():
            f.write(f"• {k.replace('_', ' ').title():<32}: {v}\n")
        f.write("\n")
        
        f.write("3. GENERATED OUTPUT ARTIFACTS\n")
        f.write("-" * 80 + "\n")
        f.write("• antenna_layout.png         : High-resolution 2D engineering layout with dimensions & stackup\n")
        f.write("• antenna_3d_view.png        : 3D isometric perspective rendering\n")
        f.write("• antenna_3d_interactive.html: Interactive WebGL/Three.js 3D model with orbital controls & HUD\n")
        f.write("• s11_reflection.png         : S11 Return Loss curve & wideband frequency response\n")
        f.write("• input_impedance.png        : Complex input impedance Zin (resistance & reactance)\n")
        if len(s_coupled_db) > 0:
            f.write("• mutual_coupling.png        : Multi-channel inter-port isolation & coupling curves\n")
        f.write("• radiation_pattern_polar.png: 2D polar far-field radiation pattern (E-plane & H-plane)\n")
        f.write("• radiation_pattern_cartesian.png: Far-field directivity vs angle plot\n")
        f.write("• s_parameters_data.csv      : Tabulated numerical frequency response data\n")
        f.write("• radiation_pattern_data.csv : Tabulated far-field radiation pattern data\n")
        f.write("• simulation_summary.json    : Structured machine-readable metrics summary\n")
        f.write("• simulation_report.txt      : Complete human-readable engineering report\n")
        f.write("=" * 80 + "\n")
        
    print(f"Exported Engineering Report: {report_path.name}")
    print("\n" + "=" * 75)
    print(f"SIMULATION & POST-PROCESSING COMPLETED SUCCESSFULLY")
    print(f"All figures, reports, and models saved in: {output_dir}")
    print("=" * 75 + "\n")

def run_simulation(antenna_type: str = 'ti_77ghz_rx',
                   engine: str = 'vulkan',
                   num_threads: int = 0,
                   vulkan_profile: bool = False,
                   vulkan_pml: str = 'fused',
                   cells_per_wavelength: float = 60.0,
                   channel: str = '1',
                   max_timesteps: int = 40000) -> pathlib.Path:
    """Execute full FDTD simulation pipeline for the specified proven antenna."""
    start_dt = datetime.datetime.now()
    timestamp_str = start_dt.strftime("%Y-%m-%d_%H-%M-%S")
    
    antenna = antenna_model.get_antenna_model(antenna_type, cells_per_wavelength=cells_per_wavelength)
    
    project_root = pathlib.Path(__file__).resolve().parent
    sim_result_root = project_root / "SimResult"
    output_dir = sim_result_root / f"{timestamp_str}_{antenna.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sim_log_path = output_dir / "simulation.log"
    logger = TeeLogger(sim_log_path)
    logger.start()
    try:
        print("=" * 75)
        print(f"STARTING SIMULATION: {antenna.title.upper()}")
        print("=" * 75)
        print(f"Start Time:           {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Reference Document:   {antenna.doc_reference}")
        print(f"Output Directory:     {output_dir}")
        print(f"Operating Band:       {antenna.f_start/1e9:.2f} GHz – {antenna.f_stop/1e9:.2f} GHz (Center: {antenna.f_center/1e9:.2f} GHz)")
        print(f"Solver Engine:        {engine}")
        print(f"Mesh Resolution:      {antenna.cells_per_wavelength:.1f} cells/lambda_0 (min >= 60)")
        print(f"Max Timesteps:        {max_timesteps}")
        print("=" * 75)
        
        # 1. Generate presentation models
        print("\n[Step 1/4] Generating visual models and layout diagrams...")
        antenna.draw_layout(str(output_dir / "antenna_layout.png"))
        antenna.draw_3d_view(str(output_dir / "antenna_3d_view.png"))
        antenna.draw_interactive_3d(str(output_dir / "antenna_3d_interactive.html"))
        print("Saved: antenna_layout.png, antenna_3d_view.png, antenna_3d_interactive.html")
        
        # 2. Build CSXCAD geometry and openEMS FDTD solver
        print("\n[Step 2/4] Building CSXCAD geometry and FDTD mesh...")
        fdtd, csx, ports, nf2ff = antenna.create_simulation(end_criteria=1e-4, nr_ts=max_timesteps, active_channels=channel)
        
        csx_xml_path = output_dir / "openEMS_model.xml"
        fdtd.Write2XML(str(csx_xml_path))
        print(f"Exported CSX/FDTD model XML: {csx_xml_path.name}")
        
        # 3. Run solver
        print("\n[Step 3/4] Running openEMS FDTD solver...")
        sim_t0 = time.time()
        openems_log_path = output_dir / "openEMS.log"
        solver_rc = run_openems_solver(
            output_dir=output_dir,
            xml_path=csx_xml_path,
            engine=engine,
            num_threads=num_threads,
            log_path=openems_log_path,
            vulkan_profile=vulkan_profile,
            vulkan_pml=vulkan_pml
        )
        if solver_rc != 0:
            print(f"Subprocess returned {solver_rc}, executing fdtd.Run()...")
            run_kwargs = {'cleanup': True}
            if engine:
                run_kwargs['engine'] = engine
            if engine in ('vulkan', 'gpu'):
                run_kwargs['vulkan_fd'] = 'gpu'
                run_kwargs['vulkan_pml'] = vulkan_pml
                if vulkan_profile:
                    run_kwargs['vulkan_profile'] = True
            if num_threads > 0:
                run_kwargs['numThreads'] = num_threads
            fdtd.Run(str(output_dir), **run_kwargs)
            
        sim_duration = time.time() - sim_t0
        print(f"Solver completed in {sim_duration:.2f} seconds.")
        
        # 4. Post-processing
        print("\n[Step 4/4] Executing post-processing and generating reports...")
        post_process_directory(
            output_dir=output_dir,
            antenna=antenna,
            sim_duration=sim_duration,
            start_dt=start_dt,
            engine=engine,
            channel=channel,
            max_timesteps=max_timesteps
        )
    finally:
        logger.stop()
        
    return output_dir

if __name__ == '__main__':
    run_simulation()
