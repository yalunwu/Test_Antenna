"""
77 GHz 4-Channel Patch Antenna Array Simulation Script
Simulates the antenna with openEMS FDTD from 76 GHz to 78 GHz.
Saves all outputs, figures, and data into a timestamped subfolder of SimResult/.
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
from antenna_model import PatchAntennaArray77GHz

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
                       log_path: pathlib.Path = None) -> int:
    """Execute openEMS solver executable while streaming and logging all output."""
    openems_exe = openems_config.get_openems_path() / 'openEMS.exe'
    
    if not openems_exe.exists():
        return -1
        
    cmd = [str(openems_exe), str(xml_path)]
    if engine:
        cmd.append(f'--engine={engine}')
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
                           engine: str = 'vulkan',
                           channel: str = '1',
                           max_timesteps: int = 50000) -> None:
    """Perform full post-processing, figure plotting, data export, and report generation."""
    if antenna is None:
        antenna = PatchAntennaArray77GHz()
    if start_dt is None:
        start_dt = datetime.datetime.now()
        
    sim_log_path = output_dir / "simulation.log"
    logger = None
    if not isinstance(sys.stdout, TeeLogger):
        logger = TeeLogger(sim_log_path)
        logger.start()

    is_all_channels = str(channel).strip().lower() in ('all', 'array')
    if not is_all_channels:
        try:
            ch_active = int(channel)
        except Exception:
            ch_active = 1
        if ch_active < 1 or ch_active > antenna.num_channels:
            ch_active = 1
        mode_str = f"Channel {ch_active} Excited (Remaining Channels Terminated)"
    else:
        ch_active = 1
        mode_str = "All 4 Channels Driven Simultaneously (Active Broadside Beamforming Array)"

    print("\n" + "=" * 70)
    print("POST-PROCESSING 4-CHANNEL ARRAY SIMULATION RESULTS")
    print(f"Directory:           {output_dir}")
    print(f"Excitation Mode:     {mode_str}")
    print(f"Max Timesteps:       {max_timesteps}")
    print("=" * 70)
    
    # Re-create ports and nf2ff wrappers
    fdtd, csx, ports, nf2ff = antenna.create_simulation(end_criteria=1e-4, nr_ts=max_timesteps, active_channels=channel)
    if not isinstance(ports, list):
        ports = [ports]
        
    # Generate antenna visual plots (2D layout, 3D isometric structure, interactive 3D HTML)
    print("Generating antenna visual models and layout diagrams...")
    antenna.draw_layout(str(output_dir / "antenna_layout.png"))
    antenna.draw_3d_view(str(output_dir / "antenna_3d_view.png"))
    antenna.draw_interactive_3d(str(output_dir / "antenna_3d_interactive.html"))
    print("Saved: antenna_layout.png, antenna_3d_view.png, antenna_3d_interactive.html")
    
    # 1. Post-Processing: S-Parameters & Impedance
    print(f"Analyzing multi-port signals, S-parameters ({mode_str})...")
    f_band = np.linspace(76.0e9, 78.0e9, 201)  # Target 76 - 78 GHz band
    f_wide = np.linspace(74.0e9, 80.0e9, 301)  # Wideband context
    
    s_coupled_db = {}
    s_active_all_band_db = {}
    s_active_all_wide_db = {}
    zin_all = {}
    
    if not is_all_channels:
        # Single active channel
        port_active = ports[ch_active - 1]
        port_active.CalcPort(str(output_dir), f_wide)
        s11_wide = port_active.uf_ref / port_active.uf_inc
        s11_wide_db = 20.0 * np.log10(np.abs(s11_wide))
        
        port_active.CalcPort(str(output_dir), f_band)
        s11_band = port_active.uf_ref / port_active.uf_inc
        s11_band_db = 20.0 * np.log10(np.abs(s11_band))
        zin_band = port_active.uf_tot / port_active.if_tot
        
        # Calculate mutual coupling S_{j, ch_active} for all secondary ports
        for ch_idx in range(len(ports)):
            if ch_idx == (ch_active - 1):
                continue
            p_sec = ports[ch_idx]
            p_sec.CalcPort(str(output_dir), f_band)
            s_kj = p_sec.uf_tot / port_active.uf_inc
            s_kj_db = 20.0 * np.log10(np.maximum(np.abs(s_kj), 1e-6))
            s_coupled_db[f'S{ch_idx+1}{ch_active}'] = s_kj_db
    else:
        # All 4 channels driven simultaneously
        for ch_idx in range(len(ports)):
            p = ports[ch_idx]
            p.CalcPort(str(output_dir), f_wide)
            s_w = p.uf_ref / p.uf_inc
            s_active_all_wide_db[f'Port {ch_idx+1}'] = 20.0 * np.log10(np.abs(s_w))
            
            p.CalcPort(str(output_dir), f_band)
            s_b = p.uf_ref / p.uf_inc
            s_active_all_band_db[f'Port {ch_idx+1}'] = 20.0 * np.log10(np.abs(s_b))
            zin_all[f'Port {ch_idx+1}'] = p.uf_tot / p.if_tot
            
        # Use Port 1 as representative reference
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
    
    idx_77 = np.argmin(np.abs(f_band - 77.0e9))
    s11_77_db = float(s11_band_db[idx_77])
    zin_77 = zin_band[idx_77]
    
    # Coupling / active reflection levels at 77.0 GHz
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
    
    if not is_all_channels:
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
    else:
        # With all channels excited in-phase, openEMS directly simulates full-wave broadside array beamforming!
        dmax_array_dbi = dmax_dbi
        hpbw_array_h = hpbw_h
        dir_array_h_dbi = dir_pattern_h_dbi
    
    # 4. Generate and Save Figures
    print("Plotting figures...")
    
    # 4. Generate and Save Figures
    print("Plotting figures...")
    
    # Figure 1: S11 Return Loss / Active Reflection
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), tight_layout=True)
    if not is_all_channels:
        ax1.plot(f_band / 1e9, s11_band_db, 'b-', linewidth=2.5, label=f'|S{ch_active}{ch_active}| (Port {ch_active} Driven)')
        ax1.axhline(-10.0, color='r', linestyle='--', linewidth=1.5, label='-10 dB Specification')
        ax1.axvline(77.0, color='gray', linestyle=':', linewidth=1.2, label='77.0 GHz Center')
        ax1.plot(f_res_ghz, min_s11_db, 'ro', markersize=8, label=f'Min S{ch_active}{ch_active}: {min_s11_db:.2f} dB @ {f_res_ghz:.2f} GHz')
        ax1.fill_between(f_band / 1e9, -40, -10, where=(s11_band_db <= -10), color='green', alpha=0.15, label='Matched Band')
        ax1.set_title(f'77 GHz 4-Channel Array — Port {ch_active} Return Loss (S{ch_active}{ch_active})', fontsize=12, fontweight='bold')
    else:
        colors_all = ['#2563eb', '#16a34a', '#ea580c', '#d9381e']
        for idx_ch, (p_name, p_sdb) in enumerate(s_active_all_band_db.items()):
            ax1.plot(f_band / 1e9, p_sdb, linewidth=2.2, color=colors_all[idx_ch % 4], label=f'{p_name} Active $\\Gamma$ ({p_sdb[idx_77]:.1f} dB @ 77 GHz)')
        ax1.axhline(-10.0, color='r', linestyle='--', linewidth=1.5, label='-10 dB Specification')
        ax1.axvline(77.0, color='gray', linestyle=':', linewidth=1.2, label='77.0 GHz Center')
        ax1.set_title('77 GHz 4-Channel Array — Active Reflection Coefficients (All Ports Driven In-Phase)', fontsize=12, fontweight='bold')
    ax1.set_xlim([76.0, 78.0])
    ax1.set_ylim([max(-35.0, np.floor(min_s11_db / 5) * 5 - 5), 0.0])
    ax1.set_xlabel('Frequency (GHz)', fontsize=11)
    ax1.set_ylabel('|S-Parameter| (dB)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=9)
    
    if not is_all_channels:
        ax2.plot(f_wide / 1e9, s11_wide_db, 'k-', linewidth=1.8, label=f'|S{ch_active}{ch_active}| (Wideband)')
    else:
        ax2.plot(f_wide / 1e9, s_active_all_wide_db['Port 1'], 'b-', linewidth=1.8, label='Port 1 (Outer) Wideband')
        ax2.plot(f_wide / 1e9, s_active_all_wide_db['Port 2'], 'g--', linewidth=1.8, label='Port 2 (Inner) Wideband')
    ax2.axvspan(76.0, 78.0, color='blue', alpha=0.1, label='Radar Band (76 - 78 GHz)')
    ax2.axhline(-10.0, color='r', linestyle='--', linewidth=1.2)
    ax2.set_xlim([74.0, 80.0])
    ax2.set_ylim([-30.0, 0.0])
    ax2.set_xlabel('Frequency (GHz)', fontsize=11)
    ax2.set_ylabel('|S-Parameter| (dB)', fontsize=11)
    ax2.set_title('Wideband Overview (74 - 80 GHz)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='lower left', framealpha=0.9, fontsize=9)
    
    s11_fig_path = output_dir / "s11_reflection.png"
    fig.savefig(s11_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {s11_fig_path.name}")
    
    # Figure 2: Mutual Coupling / Active Isolation
    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
    colors = ['#d9381e', '#ea580c', '#16a34a', '#2563eb']
    if not is_all_channels:
        for idx, (param_name, s_db) in enumerate(s_coupled_db.items()):
            target_port = int(param_name[1])
            dist_mm = abs(target_port - ch_active) * antenna.channel_spacing
            ax.plot(f_band / 1e9, s_db, linewidth=2.2, color=colors[idx % len(colors)],
                    label=f'|{param_name}| (Port {target_port}, $\\Delta x = {dist_mm:.1f}$ mm): {s_db[idx_77]:.1f} dB @ 77 GHz')
        ax.set_title(f'77 GHz 4-Channel Array — Inter-Channel Coupling (Port {ch_active} Driven)', fontsize=12, fontweight='bold')
    else:
        for idx, (param_name, s_db) in enumerate(s_active_all_band_db.items()):
            ax.plot(f_band / 1e9, s_db, linewidth=2.2, color=colors[idx % len(colors)],
                    label=f'{param_name} Active $\\Gamma$: {s_db[idx_77]:.1f} dB @ 77 GHz')
        ax.set_title('77 GHz 4-Channel Array — Active Channel Reflections Under Full Array Excitation', fontsize=12, fontweight='bold')
    ax.axhline(-25.0, color='gray', linestyle='--', linewidth=1.2, label='-25 dB MIMO Isolation Benchmark')
    ax.axvline(77.0, color='purple', linestyle=':', linewidth=1.2, label='77.0 GHz')
    ax.set_xlim([76.0, 78.0])
    ax.set_ylim([-60.0, -15.0])
    ax.set_xlabel('Frequency (GHz)', fontsize=11)
    ax.set_ylabel('Transmission / Reflection (dB)', fontsize=11)
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
    port_label = f"Port {ch_active}" if not is_all_channels else "Port 1 (Outer Active)"
    ax.set_title(f'77 GHz Array — {port_label} Input Impedance Zin', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    zin_fig_path = output_dir / "input_impedance.png"
    fig.savefig(zin_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {zin_fig_path.name}")
    
    # Figure 4: Radiation Pattern - Polar Plot
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={'projection': 'polar'}, tight_layout=True)
    theta_rad_plot = np.deg2rad(theta)
    r_min = -25.0
    r_max = 20.0
    
    pat_e_clipped = np.clip(dir_pattern_e_dbi, r_min, r_max)
    pat_h_clipped = np.clip(dir_pattern_h_dbi, r_min, r_max)
    pat_array_clipped = np.clip(dir_array_h_dbi, r_min, r_max)
    
    if not is_all_channels:
        ax.plot(theta_rad_plot, pat_e_clipped, 'b-', linewidth=2.0, label=f'Ch {ch_active} E-Plane (Elevation, HPBW={hpbw_e:.1f}°)')
        ax.plot(theta_rad_plot, pat_h_clipped, 'orange', linestyle='--', linewidth=1.8, label=f'Ch {ch_active} H-Plane (Azimuth, HPBW={hpbw_h:.1f}°)')
        ax.plot(theta_rad_plot, pat_array_clipped, 'red', linewidth=2.5, label=f'4-Ch Synthesized Broadside Beam (HPBW={hpbw_array_h:.1f}°, Peak={dmax_array_dbi:.2f} dBi)')
        title_polar = f'77 GHz Radiation Pattern (Channel {ch_active} Active)\nSynthesized Broadside Directivity = {dmax_array_dbi:.2f} dBi (Element: {dmax_dbi:.2f} dBi)'
    else:
        ax.plot(theta_rad_plot, pat_e_clipped, 'b-', linewidth=2.0, label=f'Full-Wave Array Elevation (E-Plane, HPBW={hpbw_e:.1f}°)')
        ax.plot(theta_rad_plot, pat_h_clipped, 'red', linewidth=2.5, label=f'Full-Wave Array Broadside Beam (Azimuth, HPBW={hpbw_h:.1f}°, Dmax={dmax_dbi:.2f} dBi)')
        title_polar = f'77 GHz Full-Wave 4-Channel Active Array Pattern (All Ports Driven In-Phase)\nPeak Directivity = {dmax_dbi:.2f} dBi | Azimuth HPBW = {hpbw_h:.1f}°'
    
    ax.set_theta_zero_location('N')  # 0 deg (broadside +z) at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_rlim(r_min, r_max)
    ax.set_rticks(np.arange(r_min, r_max + 1, 10))
    ax.set_rlabel_position(45)
    ax.set_title(title_polar, fontsize=11.5, fontweight='bold', pad=22)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), framealpha=0.9, fontsize=9.0)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    polar_fig_path = output_dir / "radiation_pattern_polar.png"
    fig.savefig(polar_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {polar_fig_path.name}")
    
    # Figure 5: Radiation Pattern - Cartesian Plot
    fig, ax = plt.subplots(figsize=(10, 5.5), tight_layout=True)
    if not is_all_channels:
        ax.plot(theta, dir_pattern_e_dbi, 'b-', linewidth=2.0, label=f'Ch {ch_active} E-Plane (Elevation, HPBW={hpbw_e:.1f}°)')
        ax.plot(theta, dir_pattern_h_dbi, 'orange', linestyle='--', linewidth=1.8, label=f'Ch {ch_active} H-Plane (Azimuth, HPBW={hpbw_h:.1f}°)')
        ax.plot(theta, dir_array_h_dbi, 'red', linewidth=2.5, label=f'4-Ch Synthesized Array (Azimuth, HPBW={hpbw_array_h:.1f}°, Peak={dmax_array_dbi:.2f} dBi)')
        ax.set_title(f'77 GHz Far-Field Directivity vs Angle (Channel {ch_active} Active vs 4-Ch Synthesized Beam)', fontsize=12, fontweight='bold')
    else:
        ax.plot(theta, dir_pattern_e_dbi, 'b-', linewidth=2.0, label=f'Full-Wave Array Elevation (E-Plane, HPBW={hpbw_e:.1f}°)')
        ax.plot(theta, dir_pattern_h_dbi, 'red', linewidth=2.5, label=f'Full-Wave Array Azimuth (H-Plane, HPBW={hpbw_h:.1f}°, Peak={dmax_dbi:.2f} dBi)')
        ax.set_title('77 GHz Far-Field Directivity vs Angle — Full-Wave 4-Channel Active Array', fontsize=12, fontweight='bold')
    ax.axhline(dmax_array_dbi, color='green', linestyle=':', linewidth=1.2, label=f'Array Peak: {dmax_array_dbi:.2f} dBi')
    ax.set_xlim([-180.0, 180.0])
    ax.set_ylim([-30.0, 20.0])
    ax.set_xlabel('Angle $\\theta$ (degrees)', fontsize=11)
    ax.set_ylabel('Directivity (dBi)', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    
    cart_fig_path = output_dir / "radiation_pattern_cartesian.png"
    fig.savefig(cart_fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {cart_fig_path.name}")
    
    # 5. Export CSV Data
    csv_s11_path = output_dir / "s_parameters_data.csv"
    with open(csv_s11_path, 'w', encoding='utf-8') as f:
        if not is_all_channels:
            headers = ["frequency_ghz", f"s{ch_active}{ch_active}_db"] + [f"{k.lower()}_db" for k in s_coupled_db.keys()] + ["zin_real_ohm", "zin_imag_ohm"]
            f.write(",".join(headers) + "\n")
            coupling_keys = list(s_coupled_db.keys())
            for idx_f in range(len(f_band)):
                f_val = f_band[idx_f] / 1e9
                row = [f"{f_val:.4f}", f"{s11_band_db[idx_f]:.4f}"]
                for k in coupling_keys:
                    row.append(f"{s_coupled_db[k][idx_f]:.4f}")
                row.extend([f"{zin_band[idx_f].real:.4f}", f"{zin_band[idx_f].imag:.4f}"])
                f.write(",".join(row) + "\n")
        else:
            headers = ["frequency_ghz", "port1_gamma_db", "port2_gamma_db", "port3_gamma_db", "port4_gamma_db", "port1_zin_real", "port1_zin_imag"]
            f.write(",".join(headers) + "\n")
            for idx_f in range(len(f_band)):
                f_val = f_band[idx_f] / 1e9
                row = [f"{f_val:.4f}"] + [f"{s_active_all_band_db[f'Port {p}'][idx_f]:.4f}" for p in range(1, 5)] + [f"{zin_all['Port 1'][idx_f].real:.4f}", f"{zin_all['Port 1'][idx_f].imag:.4f}"]
                f.write(",".join(row) + "\n")
    print(f"Exported CSV data: {csv_s11_path.name}")
    
    csv_pattern_path = output_dir / "radiation_pattern_77ghz.csv"
    with open(csv_pattern_path, 'w', encoding='utf-8') as f:
        f.write("theta_deg,h_plane_dbi,e_plane_dbi,array_broadside_dbi\n")
        for th_val, dh_val, de_val, da_val in zip(theta, dir_pattern_h_dbi, dir_pattern_e_dbi, dir_array_h_dbi):
            f.write(f"{th_val:.1f},{dh_val:.4f},{de_val:.4f},{da_val:.4f}\n")
    print(f"Exported CSV data: {csv_pattern_path.name}")
    
    # 6. Summary and Reports
    end_dt = datetime.datetime.now()
    summary = {
        "simulation_info": {
            "antenna_type": f"{antenna.num_channels}-Channel 3-Element Series-Fed Patch Array",
            "num_channels": antenna.num_channels,
            "active_excitation_mode": mode_str,
            "max_timesteps": max_timesteps,
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
        "inter_channel_metrics_at_77ghz": {
            k: round(v, 2) for k, v in coupling_77_db.items()
        },
        "impedance_results": {
            "zin_at_resonance": f"{zin_at_res.real:.2f} + {zin_at_res.imag:.2f}j Ohm",
            "zin_at_77ghz": f"{zin_77.real:.2f} + {zin_77.imag:.2f}j Ohm",
            "system_impedance_ohm": 50.0
        },
        "radiation_at_77ghz": {
            "peak_directivity_dbi": round(dmax_dbi, 2),
            "peak_directivity_linear": round(dmax_linear, 2),
            "hpbw_e_plane_deg": round(hpbw_e, 1),
            "hpbw_h_plane_deg": round(hpbw_h, 1),
            "front_to_back_ratio_db": round(fb_ratio_db, 2),
            "array_peak_directivity_dbi": round(dmax_array_dbi, 2),
            "array_azimuth_hpbw_deg": round(hpbw_array_h, 1)
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
        
    coupling_report_str = "\n".join([f"  - {k}: {v:.2f} dB" for k, v in coupling_77_db.items()])
    
    report_txt_path = output_dir / "simulation_report.txt"
    report_content = f"""================================================================================
77 GHz 4-CHANNEL PATCH ANTENNA ARRAY SIMULATION REPORT (4-LAYER STACKUP)
===============================================================================
Timestamp Started:       {start_dt.strftime("%Y-%m-%d %H:%M:%S")}
Timestamp Finished:      {end_dt.strftime("%Y-%m-%d %H:%M:%S")}
Total Execution Time:    {sim_duration:.2f} seconds
Output Directory:        {output_dir}
Excitation Mode:         {mode_str}
Max Timesteps:           {max_timesteps}

================================================================================
ARRAY ARCHITECTURE & CONFIGURATION
================================================================================
Number of Channels:           {antenna.num_channels} parallel series-fed patch arrays
Channel Spacing (center-to-center): {antenna.channel_spacing:.2f} mm (~{antenna.channel_spacing/antenna.lambda0_mm:.2f} lambda_0)
Channel Offsets (X-axis):     {antenna.channel_offsets_x} mm
Excitation Architecture:      {mode_str}
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

INTER-CHANNEL / ACTIVE REFLECTION LEVELS AT 77.0 GHz:
{coupling_report_str}

FAR-FIELD RADIATION AT 77.0 GHz:
Active Radiation Element / Array:
  - Peak Directivity (D_max):     {dmax_dbi:.2f} dBi ({dmax_linear:.2f} linear)
  - E-Plane HPBW (yz, Elevation): {hpbw_e:.1f} degrees
  - H-Plane HPBW (xz, Azimuth):   {hpbw_h:.1f} degrees
  - Front-to-Back Ratio:          {fb_ratio_db:.2f} dB
  - 4-Channel Array Directivity:  {dmax_array_dbi:.2f} dBi (Azimuth HPBW: {hpbw_array_h:.1f} deg)

================================================================================
GENERATED OUTPUT FILES
================================================================================
1. antenna_layout.png             - 2D layout geometry showing all 4 channels and 4-layer stackup
2. antenna_3d_view.png             - 3D isometric perspective structural plot
3. antenna_3d_interactive.html     - Interactive 3D WebGL model (orbit/zoom/pan in browser)
4. s11_reflection.png             - S11 reflection coefficient vs frequency (76-78 GHz)
5. mutual_coupling.png            - Inter-channel mutual coupling / active reflection curves
6. input_impedance.png            - Complex input impedance Zin vs frequency
7. radiation_pattern_polar.png    - 2D polar directivity patterns
8. radiation_pattern_cartesian.png- Cartesian directivity vs elevation angle
9. s_parameters_data.csv          - Numerical S-parameter matrix
10. radiation_pattern_77ghz.csv   - Numerical far-field directivity table
11. openEMS_model.xml             - CSXCAD full 3D simulation model
12. openEMS.log                   - Raw openEMS FDTD solver engine log (timesteps, speed, energy)
13. simulation.log                - Python execution log with timestamps and detailed metrics
14. simulation_summary.json       - Structured JSON results
15. simulation_report.txt         - This summary report
================================================================================
"""
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("\n" + "=" * 70)
    print("4-CHANNEL ARRAY SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Excitation Mode:          {mode_str}")
    print(f"Resonant Frequency:       {f_res_ghz:.3f} GHz")
    print(f"Return Loss S11:          {s11_77_db:.2f} dB")
    for k, v in coupling_77_db.items():
        print(f"{k} @ 77 GHz:         {v:.2f} dB")
    print(f"Directivity Dmax:         {dmax_dbi:.2f} dBi")
    print(f"Array Broadside Dmax:     {dmax_array_dbi:.2f} dBi (HPBW: {hpbw_array_h:.1f} deg)")
    print(f"Results saved in:         {output_dir}")
    print("=" * 70)
    if logger:
        logger.stop()

def run_simulation(engine: str = 'vulkan',
                   num_threads: int = 0,
                   cells_per_wavelength: float = 40.0,
                   channel: str = '1',
                   max_timesteps: int = 50000) -> pathlib.Path:
    # 1. Record start time and create timestamped directory
    start_dt = datetime.datetime.now()
    timestamp_str = start_dt.strftime("%Y-%m-%d_%H-%M-%S")
    
    project_root = pathlib.Path(__file__).resolve().parent
    sim_result_root = project_root / "SimResult"
    output_dir = sim_result_root / timestamp_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sim_log_path = output_dir / "simulation.log"
    logger = TeeLogger(sim_log_path)
    logger.start()
    try:
        is_all_channels = str(channel).strip().lower() in ('all', 'array')
        mode_str = "All 4 Channels (Simultaneous In-Phase Array Excitation)" if is_all_channels else f"Channel {channel} Driven (Remaining Channels Terminated)"
        print("=" * 70)
        print("77 GHz 4-CHANNEL PATCH ANTENNA ARRAY SIMULATION")
        print("=" * 70)
        print(f"Simulation Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output Directory:     {output_dir}")
        print(f"Log Files:            openEMS.log & simulation.log")
        print(f"Excitation Mode:      {mode_str}")
        print(f"Max Timesteps:        {max_timesteps} (>= 3x excitation length)")
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
        fdtd, csx, ports, nf2ff = antenna.create_simulation(end_criteria=1e-4, nr_ts=max_timesteps, active_channels=channel)
        
        # Export CSX model XML with full FDTD setup
        csx_xml_path = output_dir / "openEMS_model.xml"
        fdtd.Write2XML(str(csx_xml_path))
        print(f"Exported CSX/FDTD model XML: {csx_xml_path.name}")
        
        # 4. Execute FDTD Simulation
        print("\nStarting openEMS FDTD solver...")
        sim_t0 = time.time()
        openems_log_path = output_dir / "openEMS.log"
        solver_rc = run_openems_solver(
            output_dir=output_dir,
            xml_path=csx_xml_path,
            engine=engine,
            num_threads=num_threads,
            log_path=openems_log_path
        )
        if solver_rc != 0:
            print(f"Warning: openEMS subprocess returned code {solver_rc}, falling back to fdtd.Run()...")
            run_kwargs = {'cleanup': True}
            if engine:
                run_kwargs['engine'] = engine
            if num_threads > 0:
                run_kwargs['numThreads'] = num_threads
            fdtd.Run(str(output_dir), **run_kwargs)
            
        sim_duration = time.time() - sim_t0
        print(f"FDTD simulation completed in {sim_duration:.2f} seconds.")
        print(f"openEMS solver log saved to: {openems_log_path.name}")
        print(f"Python simulation log saved to: {sim_log_path.name}")
        
        # 5. Execute Post-Processing
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
