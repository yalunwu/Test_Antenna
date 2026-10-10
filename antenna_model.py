"""
77 GHz Patch Antenna Array Model for Automotive Radar
Designed for openEMS FDTD Simulation.
Standard 4-Layer Automotive mmWave Radar Stackup:
  - Layer 1 (Top RF): Microstrip Patches & 50-Ohm Feed Line (Copper)
  - Dielectric 1: 0.127 mm (5 mil) Rogers RO3003 (eps_r = 3.00, tan_d = 0.0010)
  - Layer 2: Continuous RF Ground Plane (Copper at z = 0)
  - Dielectric 2: 0.800 mm (31.5 mil) FR-4 Structural Core (eps_r = 4.40, tan_d = 0.020)
  - Layer 4 (Bottom): Continuous Shielding Ground Plane (Copper at z = -0.80 mm)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import environment config first
import openems_config

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

class PatchAntennaArray77GHz:
    def __init__(self,
                 f_center: float = 77.0e9,
                 f_start: float = 76.0e9,
                 f_stop: float = 78.0e9,
                 num_channels: int = 4,         # 4 channels side-by-side
                 channel_spacing: float = 5.000,# mm (center-to-center channel spacing along x)
                 num_elements: int = 3,         # 3 series-fed patches per channel
                 patch_length: float = 1.070,   # mm (resonant length along y)
                 patch_width: float = 1.350,    # mm (resonant width along x)
                 feed_width: float = 0.320,     # mm (50 Ohm line width)
                 feed_length: float = 1.000,    # mm
                 trans_width: float = 0.100,    # mm (quarter-wave transformer width ~95 Ohm)
                 trans_length: float = 0.615,   # mm (quarter-wave transformer length)
                 connect_width: float = 0.150,  # mm (connecting line width between patches)
                 connect_length: float = 1.340, # mm (connecting line length)
                 sub_h: float = 0.127,          # mm (Rogers RO3003 thickness)
                 sub_eps_r: float = 3.00,       # Rogers RO3003 dielectric constant
                 sub_tan_d: float = 0.0010,     # loss tangent
                 sub_margin_x: float = 3.000,   # mm margin around array in x
                 sub_margin_y: float = 1.500,   # mm margin around array in y
                 h_core: float = 0.800,         # mm (FR-4 core thickness in 4-layer stackup)
                 eps_core: float = 4.40,        # FR-4 core dielectric constant
                 tan_d_core: float = 0.020,     # FR-4 core loss tangent
                 is_4layer: bool = True,        # Enable realistic 4-layer stackup
                 cells_per_wavelength: float = 40.0  # Mesh resolution (cells per lambda_0)
                 ):
        self.f_center = f_center
        self.f_start = f_start
        self.f_stop = f_stop
        self.num_channels = num_channels
        self.channel_spacing = channel_spacing
        self.num_elements = num_elements
        self.cells_per_wavelength = cells_per_wavelength
        
        # Channel offsets along x (centered around x = 0)
        self.channel_offsets_x = [
            (i - (self.num_channels - 1) / 2.0) * self.channel_spacing
            for i in range(self.num_channels)
        ]
        
        # Dimensions in millimeters
        self.patch_length = patch_length
        self.patch_width = patch_width
        self.feed_width = feed_width
        self.feed_length = feed_length
        self.trans_width = trans_width
        self.trans_length = trans_length
        self.connect_width = connect_width
        self.connect_length = connect_length
        
        # Substrate properties (Dielectric 1 - RF Cavity)
        self.sub_h = sub_h
        self.sub_eps_r = sub_eps_r
        self.sub_tan_d = sub_tan_d
        self.sub_kappa = 2 * np.pi * self.f_center * EPS0 * self.sub_eps_r * self.sub_tan_d
        
        # 4-Layer Mechanical Core properties (Dielectric 2 - Structural Carrier)
        self.h_core = h_core
        self.eps_core = eps_core
        self.tan_d_core = tan_d_core
        self.kappa_core = 2 * np.pi * self.f_center * EPS0 * self.eps_core * self.tan_d_core
        self.is_4layer = is_4layer
        
        # Free space wavelength in mm
        self.unit = 1e-3  # coordinate unit = 1 mm
        self.lambda0_mm = (C0 / self.f_center) / self.unit
        
        # Array coordinate calculations
        self.sub_margin_x = sub_margin_x
        self.sub_margin_y = sub_margin_y
        
        # Build coordinates along y
        # y = 0 is port plane
        self.y_feed_start = 0.0
        self.y_feed_end = self.feed_length
        
        self.y_trans_start = self.y_feed_end
        self.y_trans_end = self.y_trans_start + self.trans_length
        
        self.patch_positions = []
        self.connect_positions = []
        
        curr_y = self.y_trans_end
        for i in range(self.num_elements):
            p_start = curr_y
            p_end = p_start + self.patch_length
            self.patch_positions.append((p_start, p_end))
            curr_y = p_end
            if i < self.num_elements - 1:
                c_start = curr_y
                c_end = c_start + self.connect_length
                self.connect_positions.append((c_start, c_end))
                curr_y = c_end
        
        self.total_array_length = curr_y
        
        # Substrate dimensions (spanning all channels + margins)
        max_x = max(abs(x) for x in self.channel_offsets_x)
        self.sub_w = max(2.0 * (max_x + self.patch_width / 2.0 + self.sub_margin_x), 8.0)
        self.sub_y_min = -self.sub_margin_y
        self.sub_y_max = self.total_array_length + self.sub_margin_y
        self.sub_l = self.sub_y_max - self.sub_y_min

    def create_simulation(self, end_criteria: float = 1e-4, nr_ts: int = 50000, active_channels=1):
        """
        Instantiate CSXCAD geometry and openEMS FDTD solver for the 4-channel array.
        
        Parameters
        ----------
        end_criteria : float
            FDTD energy decay termination criteria (default: 1e-4 = -40 dB).
        nr_ts : int
            Maximum number of FDTD timesteps (default: 50000, ensures >= 3x excitation length).
        active_channels : int, str, or list
            Which channel(s) are actively excited:
            - 1, 2, 3, or 4: Single channel driven, remaining channels terminated in 50 Ohm.
            - 'all' or [1, 2, 3, 4]: All 4 channels simultaneously driven in-phase.
        """
        if active_channels is None:
            self.active_channels = {1}
        elif isinstance(active_channels, str) and active_channels.strip().lower() in ('all', 'array'):
            self.active_channels = set(range(1, self.num_channels + 1))
        elif isinstance(active_channels, (list, tuple, set)):
            self.active_channels = {int(c) for c in active_channels}
        else:
            try:
                ch = int(active_channels)
                self.active_channels = {ch} if 1 <= ch <= self.num_channels else {1}
            except Exception:
                self.active_channels = {1}

        FDTD = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
        
        # Excitation pulse covering band
        fc = 4.0e9  # corner frequency (bandwidth)
        FDTD.SetGaussExcite(self.f_center, fc)
        
        # 8-cell Perfectly Matched Layer on all boundaries
        FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(self.unit)
        
        # Add materials
        substrate = CSX.AddMaterial('RO3003', epsilon=self.sub_eps_r, kappa=self.sub_kappa)
        if self.is_4layer:
            fr4 = CSX.AddMaterial('FR4_Core', epsilon=self.eps_core, kappa=self.kappa_core)
        copper = CSX.AddMetal('Copper')
        
        # 1. Dielectric 1: Rogers RO3003 box (z from 0 to sub_h)
        sub_start = [-self.sub_w / 2.0, self.sub_y_min, 0.0]
        sub_stop = [self.sub_w / 2.0, self.sub_y_max, self.sub_h]
        substrate.AddBox(start=sub_start, stop=sub_stop, priority=0)
        
        # 2. Layer 2: Solid RF Ground Plane at z = 0
        gnd_start = [-self.sub_w / 2.0, self.sub_y_min, 0.0]
        gnd_stop = [self.sub_w / 2.0, self.sub_y_max, 0.0]
        copper.AddBox(start=gnd_start, stop=gnd_stop, priority=10)
        
        if self.is_4layer:
            # 3. Dielectric 2: FR-4 Mechanical Core (z from -h_core to 0)
            fr4_start = [-self.sub_w / 2.0, self.sub_y_min, -self.h_core]
            fr4_stop = [self.sub_w / 2.0, self.sub_y_max, 0.0]
            fr4.AddBox(start=fr4_start, stop=fr4_stop, priority=0)
            
            # 4. Layer 4: Bottom Ground / Shielding Plane at z = -h_core
            bot_start = [-self.sub_w / 2.0, self.sub_y_min, -self.h_core]
            bot_stop = [self.sub_w / 2.0, self.sub_y_max, -self.h_core]
            copper.AddBox(start=bot_start, stop=bot_stop, priority=10)
        
        # 5. Layer 1: Top Metallization across all channels at z = sub_h
        for x_center in self.channel_offsets_x:
            # Microstrip feed line (50 Ohm)
            feed_start = [x_center - self.feed_width / 2.0, self.y_feed_start, self.sub_h]
            feed_stop = [x_center + self.feed_width / 2.0, self.y_feed_end, self.sub_h]
            copper.AddBox(start=feed_start, stop=feed_stop, priority=10)
            
            # Quarter-wave transformer section
            trans_start = [x_center - self.trans_width / 2.0, self.y_trans_start, self.sub_h]
            trans_stop = [x_center + self.trans_width / 2.0, self.y_trans_end, self.sub_h]
            copper.AddBox(start=trans_start, stop=trans_stop, priority=10)
            
            # Radiating patches and connecting lines
            for idx, (p_start, p_end) in enumerate(self.patch_positions):
                p_box_start = [x_center - self.patch_width / 2.0, p_start, self.sub_h]
                p_box_stop = [x_center + self.patch_width / 2.0, p_end, self.sub_h]
                copper.AddBox(start=p_box_start, stop=p_box_stop, priority=10)
            
            for idx, (c_start, c_end) in enumerate(self.connect_positions):
                c_box_start = [x_center - self.connect_width / 2.0, c_start, self.sub_h]
                c_box_stop = [x_center + self.connect_width / 2.0, c_end, self.sub_h]
                copper.AddBox(start=c_box_start, stop=c_box_stop, priority=10)
        
        # 6. Lumped 50-Ohm Ports (Driven ports excited; others terminated in 50-Ohm matched loads)
        ports = []
        for ch_idx, x_center in enumerate(self.channel_offsets_x):
            port_nr = ch_idx + 1
            port_start = [x_center - self.feed_width / 2.0, 0.0, 0.0]
            port_stop = [x_center + self.feed_width / 2.0, 0.0, self.sub_h]
            excite_val = 1.0 if port_nr in self.active_channels else 0.0
            port = FDTD.AddLumpedPort(port_nr, 50.0, port_start, port_stop, 'z', excite_val, priority=20, edges2grid='xy')
            ports.append(port)
        
        # 7. Mesh grid definition
        res = self.lambda0_mm / self.cells_per_wavelength  # target grid resolution in free space
        
        # Scaling mesh lines based on resolution
        if self.cells_per_wavelength >= 36:
            nz_sub = 6
            nz_core = 9
            smooth_ratio = 1.30
        elif self.cells_per_wavelength >= 26:
            nz_sub = 5
            nz_core = 7
            smooth_ratio = 1.35
        else:
            nz_sub = 4
            nz_core = 5
            smooth_ratio = 1.40
        
        # Z-direction: substrate layers + air buffer
        if self.is_4layer:
            z_lines = np.concatenate([
                [-self.h_core - self.lambda0_mm / 2.0],
                np.linspace(-self.h_core, 0.0, nz_core),
                np.linspace(0.0, self.sub_h, nz_sub),
                [self.sub_h + self.lambda0_mm * 1.5]
            ])
        else:
            z_lines = np.concatenate([
                [-self.lambda0_mm / 2.0],
                np.linspace(0.0, self.sub_h, nz_sub),
                [self.sub_h + self.lambda0_mm * 1.5]
            ])
        mesh.AddLine('z', np.unique(z_lines))
        
        # X-direction: feature edges for all channels and boundaries
        x_features = [
            -self.sub_w / 2.0 - self.lambda0_mm / 3.0,
            self.sub_w / 2.0 + self.lambda0_mm / 3.0
        ]
        for x_center in self.channel_offsets_x:
            x_features.extend([
                x_center - self.patch_width / 2.0, x_center + self.patch_width / 2.0,
                x_center - self.feed_width / 2.0, x_center + self.feed_width / 2.0,
                x_center - self.trans_width / 2.0, x_center + self.trans_width / 2.0,
                x_center - self.connect_width / 2.0, x_center + self.connect_width / 2.0
            ])
        mesh.AddLine('x', x_features)
        
        # Y-direction: key feature edges and boundary
        y_features = [
            self.sub_y_min - self.lambda0_mm / 3.0,
            self.sub_y_max + self.lambda0_mm / 3.0,
            self.y_feed_start, self.y_feed_end,
            self.y_trans_start, self.y_trans_end
        ]
        for p_start, p_end in self.patch_positions:
            y_features.extend([p_start, p_end])
        for c_start, c_end in self.connect_positions:
            y_features.extend([c_start, c_end])
        mesh.AddLine('y', y_features)
        
        # Smooth mesh with ratio
        mesh.SmoothMeshLines('x', res, smooth_ratio)
        mesh.SmoothMeshLines('y', res, smooth_ratio)
        mesh.SmoothMeshLines('z', res, smooth_ratio)
        
        # 8. Near-Field to Far-Field (NF2FF) Box
        nf2ff = FDTD.CreateNF2FFBox(frequency=[self.f_center], over_sampling=4)
        
        return FDTD, CSX, ports, nf2ff

    def draw_layout(self, output_path: str):
        """Generate and save an engineering layout diagram with Top View (XY) and Side View (YZ)."""
        fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={'width_ratios': [2.4, 1]}, tight_layout=True)
        
        # --- PANEL 1: Top-down View (XY) ---
        sub_rect = mpatches.Rectangle(
            (-self.sub_w / 2.0, self.sub_y_min),
            self.sub_w, self.sub_l,
            facecolor='#e8eaed', edgecolor='#5f6368', linewidth=1.5,
            label='Rogers RO3003 Substrate'
        )
        ax_top.add_patch(sub_rect)
        
        for ch_idx, x_center in enumerate(self.channel_offsets_x):
            # Feed line
            feed_rect = mpatches.Rectangle(
                (x_center - self.feed_width / 2.0, self.y_feed_start),
                self.feed_width, self.feed_length,
                facecolor='#d9381e', edgecolor='#8a2212', linewidth=1.0,
                label='50 $\\Omega$ Feed Line' if ch_idx == 0 else ""
            )
            ax_top.add_patch(feed_rect)
            
            # Transformer
            trans_rect = mpatches.Rectangle(
                (x_center - self.trans_width / 2.0, self.y_trans_start),
                self.trans_width, self.trans_length,
                facecolor='#f9ab00', edgecolor='#b06000', linewidth=1.0,
                label='$\\lambda_g/4$ Transformer' if ch_idx == 0 else ""
            )
            ax_top.add_patch(trans_rect)
            
            # Patches
            for i, (p_start, p_end) in enumerate(self.patch_positions):
                p_rect = mpatches.Rectangle(
                    (x_center - self.patch_width / 2.0, p_start),
                    self.patch_width, self.patch_length,
                    facecolor='#1a73e8', edgecolor='#0d47a1', linewidth=1.0,
                    label='Patch Elements' if (ch_idx == 0 and i == 0) else ""
                )
                ax_top.add_patch(p_rect)
                
            # Connecting lines
            for i, (c_start, c_end) in enumerate(self.connect_positions):
                c_rect = mpatches.Rectangle(
                    (x_center - self.connect_width / 2.0, c_start),
                    self.connect_width, self.connect_length,
                    facecolor='#137333', edgecolor='#0b461f', linewidth=1.0,
                    label='Connecting Lines' if (ch_idx == 0 and i == 0) else ""
                )
                ax_top.add_patch(c_rect)
                
            # Port line & label
            ax_top.plot([x_center - self.feed_width / 2.0, x_center + self.feed_width / 2.0], [0, 0], 'r-', linewidth=3.0)
            ax_top.text(x_center, -0.35, f'Port {ch_idx+1}', ha='center', va='top',
                        color='#d9381e', fontweight='bold', fontsize=8.0)
            ax_top.text(x_center, self.total_array_length + 0.35, f'Ch #{ch_idx+1}', ha='center', va='bottom',
                        color='#1e293b', fontweight='bold', fontsize=8.5)
            
        # Draw channel spacing dimension arrows
        if len(self.channel_offsets_x) > 1:
            for k in range(len(self.channel_offsets_x) - 1):
                x_a = self.channel_offsets_x[k]
                x_b = self.channel_offsets_x[k+1]
                y_dim = self.patch_positions[1][0]
                ax_top.annotate('', xy=(x_b, y_dim), xytext=(x_a, y_dim),
                                arrowprops=dict(arrowstyle='<->', color='#334155', lw=1.2))
                ax_top.text((x_a + x_b)/2.0, y_dim + 0.25, f'{self.channel_spacing:.1f} mm',
                            ha='center', va='bottom', fontsize=8.0, color='#1e293b', fontweight='bold')
                
        ax_top.set_xlim([-self.sub_w / 2.0 - 0.8, self.sub_w / 2.0 + 0.8])
        ax_top.set_ylim([self.sub_y_min - 0.8, self.sub_y_max + 0.8])
        ax_top.set_aspect('equal')
        ax_top.set_xlabel('X Dimension (mm)', fontsize=11)
        ax_top.set_ylabel('Y Dimension (mm)', fontsize=11)
        ax_top.set_title(f'(a) Top View (XY Plane) — {self.num_channels}-Channel Array (Spacing = {self.channel_spacing:.1f} mm)',
                         fontsize=11, fontweight='bold', pad=10)
        ax_top.legend(loc='upper right', framealpha=0.9, fontsize=8.0)
        ax_top.grid(True, linestyle=':', alpha=0.6)
        
        # --- PANEL 2: Side Elevation View (YZ Cross-Section) ---
        if self.is_4layer:
            # Layer 4 Bottom Ground Plane at z = -self.h_core
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -self.h_core - 0.02), self.sub_l, 0.02,
                                                facecolor='#b87333', edgecolor='#7c2d12', linewidth=1.2, label='Layer 4 Bottom GND'))
            # Dielectric 2: FR-4 Mechanical Core from z = -self.h_core to 0
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -self.h_core), self.sub_l, self.h_core,
                                                facecolor='#dcfce7', edgecolor='#16a34a', linewidth=1.2, label='FR-4 Core (0.80 mm)'))
            ax_side.text(0, -self.h_core / 2, f'FR-4 Structural Core ({self.h_core:.2f} mm, $\\varepsilon_r=4.4$)',
                         ha='center', va='center', color='#166534', fontweight='bold', fontsize=8.5)
            
            # Layer 2: Continuous RF Ground Plane at z = 0
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -0.015), self.sub_l, 0.015,
                                                facecolor='#b87333', edgecolor='#7c2d12', linewidth=1.2, label='Layer 2 RF GND'))
            
            # Dielectric 1: Rogers RO3003 from z = 0 to sub_h
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, 0), self.sub_l, self.sub_h,
                                                facecolor='#d0e1fd', edgecolor='#4285f4', linewidth=1.5, label='RO3003 RF Cavity (0.127 mm)'))
            ax_side.text(0, self.sub_h / 2, f'Rogers RO3003 ({self.sub_h:.3f} mm, $\\varepsilon_r=3.0$)',
                         ha='center', va='center', color='#1e40af', fontweight='bold', fontsize=7.5)
        else:
            # Substrate block in YZ
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, 0), self.sub_l, self.sub_h,
                                                facecolor='#d0e1fd', edgecolor='#4285f4', linewidth=1.5, label='RO3003 Dielectric'))
            # Ground plane at z = 0
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -0.015), self.sub_l, 0.015,
                                                facecolor='#b87333', edgecolor='#7c2d12', linewidth=1.2, label='PEC Ground'))
            
        # Top metal: feed line
        ax_side.add_patch(mpatches.Rectangle((self.y_feed_start, self.sub_h), self.feed_length, 0.015,
                                            facecolor='#d9381e', edgecolor='#8a2212', linewidth=1.0))
        # Transformer
        ax_side.add_patch(mpatches.Rectangle((self.y_trans_start, self.sub_h), self.trans_length, 0.015,
                                            facecolor='#f9ab00', edgecolor='#b06000', linewidth=1.0))
        # Patches
        for i, (p_start, p_end) in enumerate(self.patch_positions):
            ax_side.add_patch(mpatches.Rectangle((p_start, self.sub_h), self.patch_length, 0.015,
                                                facecolor='#1a73e8', edgecolor='#0d47a1', linewidth=1.0, label='Layer 1 Top Cu' if i == 0 else ""))
            ax_side.text((p_start + p_end)/2, self.sub_h + 0.035, f'P#{i+1}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1a73e8')
            
        # Connecting lines
        for (c_start, c_end) in self.connect_positions:
            ax_side.add_patch(mpatches.Rectangle((c_start, self.sub_h), self.connect_length, 0.015,
                                                facecolor='#137333', edgecolor='#0b461f', linewidth=1.0))
            
        # Port arrow at y=0, z from 0 to sub_h
        ax_side.annotate('50 $\\Omega$ Port', xy=(0, self.sub_h/2), xytext=(-0.8, self.sub_h*1.8),
                         arrowprops=dict(facecolor='#d9381e', shrink=0.08, width=1.5, headwidth=6),
                         ha='center', fontsize=8.5, fontweight='bold', color='#d9381e')
        
        ax_side.set_xlim([self.sub_y_min - 0.5, self.sub_y_max + 0.5])
        if self.is_4layer:
            ax_side.set_ylim([-self.h_core - 0.15, self.sub_h * 4.0])
            title_side = f'(b) 4-Layer Automotive Stackup (Total PCB ~{(self.h_core + self.sub_h):.2f} mm)'
        else:
            ax_side.set_ylim([-0.05, self.sub_h * 3.5])
            title_side = f'(b) Cross-Section (YZ) — Stackup (h = {self.sub_h*1e3:.0f} $\\mu$m / 5 mil)'
        ax_side.set_xlabel('Y Dimension (mm)', fontsize=11)
        ax_side.set_ylabel('Z Dimension (mm)', fontsize=11)
        ax_side.set_title(title_side, fontsize=11, fontweight='bold', pad=10)
        ax_side.legend(loc='upper right', framealpha=0.9, fontsize=8.0)
        ax_side.grid(True, linestyle=':', alpha=0.6)
        
        stackup_desc = "4-Layer Automotive Radar PCB Stackup" if self.is_4layer else "Isolated RF Cavity"
        fig.suptitle(f'77 GHz {self.num_channels}-Channel Patch Antenna Array (d = {self.channel_spacing:.1f} mm)\nFull Engineering Layout & {stackup_desc}',
                     fontsize=13, fontweight='bold', y=0.98)
        
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_3d_view(self, output_path: str):
        """Generate and save a 3D isometric perspective diagram of the antenna array."""
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        
        # Dimensions
        x0, x1 = -self.sub_w / 2.0, self.sub_w / 2.0
        y0, y1 = self.sub_y_min, self.sub_y_max
        
        # Heights for 3D visual representation
        z_gnd = 0.0
        z_rf = 0.50       # visually scaled Rogers RO3003 layer
        z_core = -1.20    # visually scaled FR-4 core layer
        
        if self.is_4layer:
            # 1. Layer 4 Bottom Ground Plane at z = z_core
            bot_face = [[[x0, y0, z_core], [x1, y0, z_core], [x1, y1, z_core], [x0, y1, z_core]]]
            bot_poly = Poly3DCollection(bot_face, facecolors='#b45309', edgecolors='#78350f', alpha=0.95, linewidth=1.0)
            ax.add_collection3d(bot_poly)
            
            # 2. Dielectric 2: FR-4 Core 3D Box (z in [z_core, z_gnd])
            core_faces = [
                [[x0, y0, z_core], [x1, y0, z_core], [x1, y1, z_core], [x0, y1, z_core]],  # bottom
                [[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y1, z_gnd], [x0, y1, z_gnd]],    # top
                [[x0, y0, z_core], [x1, y0, z_core], [x1, y0, z_gnd], [x0, y0, z_gnd]],    # front
                [[x0, y1, z_core], [x1, y1, z_core], [x1, y1, z_gnd], [x0, y1, z_gnd]],    # back
                [[x0, y0, z_core], [x0, y1, z_core], [x0, y1, z_gnd], [x0, y0, z_gnd]],    # left
                [[x1, y0, z_core], [x1, y1, z_core], [x1, y1, z_gnd], [x1, y0, z_gnd]],    # right
            ]
            core_poly = Poly3DCollection(core_faces, facecolors='#86efac', edgecolors='#22c55e', alpha=0.22, linewidth=0.7)
            ax.add_collection3d(core_poly)
            ax.text(x1 + 0.3, (y0 + y1) / 2, (z_core + z_gnd) / 2, 'FR-4 Core\n(0.80 mm)',
                    fontsize=8.5, color='#15803d', fontweight='bold')
        
        # 3. Layer 2 Solid RF Ground Plane (z = 0)
        gnd_face = [[[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y1, z_gnd], [x0, y1, z_gnd]]]
        gnd_poly = Poly3DCollection(gnd_face, facecolors='#b45309', edgecolors='#78350f', alpha=0.9, linewidth=1.0)
        ax.add_collection3d(gnd_poly)
        if self.is_4layer:
            ax.text(x1 + 0.3, (y0 + y1) / 2, z_gnd, 'Layer 2 RF GND',
                    fontsize=8.5, color='#b45309', fontweight='bold')
        
        # 4. Dielectric 1: Rogers RO3003 3D Box (z in [z_gnd, z_rf])
        sub_faces = [
            [[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y1, z_gnd], [x0, y1, z_gnd]],  # bottom
            [[x0, y0, z_rf], [x1, y0, z_rf], [x1, y1, z_rf], [x0, y1, z_rf]],    # top
            [[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y0, z_rf], [x0, y0, z_rf]],    # front
            [[x0, y1, z_gnd], [x1, y1, z_gnd], [x1, y1, z_rf], [x0, y1, z_rf]],    # back
            [[x0, y0, z_gnd], [x0, y1, z_gnd], [x0, y1, z_rf], [x0, y0, z_rf]],    # left
            [[x1, y0, z_gnd], [x1, y1, z_gnd], [x1, y1, z_rf], [x1, y0, z_rf]],    # right
        ]
        sub_poly = Poly3DCollection(sub_faces, facecolors='#dbeafe', edgecolors='#93c5fd', alpha=0.35, linewidth=0.8)
        ax.add_collection3d(sub_poly)
        ax.text(x1 + 0.3, (y0 + y1) / 2, (z_gnd + z_rf) / 2, 'Rogers RO3003\n(0.127 mm)',
                fontsize=8.5, color='#1d4ed8', fontweight='bold')
        
        # 5. Layer 1 Top Copper Components across all channels (at z = z_rf)
        def add_copper_box(x_center, x_w, y_start, y_end, color):
            poly = [[[x_center - x_w/2, y_start, z_rf],
                     [x_center + x_w/2, y_start, z_rf],
                     [x_center + x_w/2, y_end, z_rf],
                     [x_center - x_w/2, y_end, z_rf]]]
            c = Poly3DCollection(poly, facecolors=color, edgecolors='#7c2d12', alpha=1.0, linewidth=1.0)
            ax.add_collection3d(c)
            return c
        
        for ch_idx, x_center in enumerate(self.channel_offsets_x):
            add_copper_box(x_center, self.feed_width, self.y_feed_start, self.y_feed_end, '#ea580c')
            add_copper_box(x_center, self.trans_width, self.y_trans_start, self.y_trans_end, '#f59e0b')
            for p_start, p_end in self.patch_positions:
                add_copper_box(x_center, self.patch_width, p_start, p_end, '#2563eb')
            for c_start, c_end in self.connect_positions:
                add_copper_box(x_center, self.connect_width, c_start, c_end, '#16a34a')
                
            # Port marker
            ax.plot([x_center, x_center], [0, 0], [z_gnd, z_rf], color='#dc2626', linewidth=3.5, marker='o', markersize=5)
            ax.text(x_center, -0.4, z_rf / 2.0, f'P{ch_idx+1}', ha='center', va='center',
                    color='#dc2626', fontweight='bold', fontsize=8.0,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor='#fee2e2', alpha=0.85, edgecolor='red'))
            ax.text(x_center, self.total_array_length + 0.3, z_rf + 0.1, f'Ch{ch_idx+1}',
                    ha='center', va='bottom', fontsize=8.0, color='#1e293b', fontweight='bold')
        
        z_min_plot = z_core - 0.3 if self.is_4layer else -0.2
        ax.set_xlim([-self.sub_w/2 - 0.8, self.sub_w/2 + 0.8])
        ax.set_ylim([self.sub_y_min - 0.5, self.sub_y_max + 0.5])
        ax.set_zlim([z_min_plot, z_rf + 0.9])
        
        ax.set_xlabel('X Dimension (mm)', fontsize=10, labelpad=8)
        ax.set_ylabel('Y Dimension (mm)', fontsize=10, labelpad=10)
        ax.set_zlabel('Z Dimension (visually scaled)', fontsize=10, labelpad=8)
        
        ax.view_init(elev=30, azim=-60)
        title_3d = f'77 GHz {self.num_channels}-Channel Patch Antenna Array (d = {self.channel_spacing:.1f} mm)\n3D Isometric Structure & Physical Model'
        ax.set_title(title_3d, fontsize=12, fontweight='bold', pad=15)
        
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_interactive_3d(self, output_path: str):
        """Generate a self-contained interactive 3D HTML visualization using Three.js."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>77 GHz 3-Element Patch Antenna — Interactive 3D Model</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #f8fafc;
        }}
        #canvas-container {{
            width: 100vw;
            height: 100vh;
        }}
        #hud {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(8px);
            padding: 16px 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            max-width: 320px;
            pointer-events: auto;
        }}
        h2 {{
            margin: 0 0 8px 0;
            font-size: 16px;
            color: #38bdf8;
            font-weight: 700;
        }}
        p {{
            margin: 4px 0;
            font-size: 12px;
            color: #94a3b8;
            line-height: 1.4;
        }}
        .metric {{
            font-weight: 600;
            color: #f1f5f9;
        }}
        .legend {{
            margin-top: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 11px;
            margin: 4px 0;
        }}
        .color-box {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-right: 8px;
        }}
        #instructions {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(15, 23, 42, 0.75);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            color: #cbd5e1;
            pointer-events: none;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
    </style>
    <!-- Three.js and OrbitControls -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="canvas-container"></div>
    <div id="hud">
        <h2>77 GHz {self.num_channels}-Channel Patch Antenna Array</h2>
        <p>Operating Band: <span class="metric">76.0 – 78.0 GHz</span></p>
        <p>Channels: <span class="metric">{self.num_channels} Parallel Arrays</span></p>
        <p>Channel Spacing: <span class="metric">{self.channel_spacing:.1f} mm (~1.3 &lambda;0)</span></p>
        <p>Stackup: <span class="metric">4-Layer Automotive Radar</span></p>
        <p>Dielectric 1 (RF): <span class="metric">Rogers RO3003 (0.127 mm)</span></p>
        <p>Layer 2 (Ground): <span class="metric">Continuous RF Copper GND</span></p>
        <p>Dielectric 2 (Core): <span class="metric">Rigid FR-4 (0.800 mm)</span></p>
        <p>Layer 4 (Back): <span class="metric">Bottom Shielding GND Plane</span></p>
        <p>Port Architecture: <span class="metric">4 Independent 50 &Omega; Ports</span></p>
        
        <div class="legend">
            <div class="legend-item"><div class="color-box" style="background:#2563eb;"></div>Patch Elements (3 per Ch)</div>
            <div class="legend-item"><div class="color-box" style="background:#f59e0b;"></div>Matching Transformer (95 &Omega;)</div>
            <div class="legend-item"><div class="color-box" style="background:#ea580c;"></div>50 &Omega; Feed Lines</div>
            <div class="legend-item"><div class="color-box" style="background:#16a34a;"></div>Connecting Lines (&lambda;g)</div>
            <div class="legend-item"><div class="color-box" style="background:#dc2626;"></div>Port 1 (Driven Source)</div>
            <div class="legend-item"><div class="color-box" style="background:#475569;"></div>Ports 2-4 (50 &Omega; Terminated)</div>
            <div class="legend-item"><div class="color-box" style="background:#93c5fd; opacity:0.6;"></div>Rogers RO3003 (0.127 mm)</div>
            <div class="legend-item"><div class="color-box" style="background:#b45309;"></div>Layer 2 RF Ground Plane</div>
            <div class="legend-item"><div class="color-box" style="background:#10b981; opacity:0.5;"></div>FR-4 Structural Core (0.80 mm)</div>
            <div class="legend-item"><div class="color-box" style="background:#78350f;"></div>Layer 4 Bottom Ground Plane</div>
        </div>
    </div>
    
    <div id="instructions">
        Rotate: Left Click + Drag | Pan: Right Click + Drag | Zoom: Scroll Wheel
    </div>

    <script>
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(16, 20, 20);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 3, 0);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
        dirLight.position.set(15, 25, 20);
        dirLight.castShadow = true;
        scene.add(dirLight);

        const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
        dirLight2.position.set(-15, -10, -10);
        scene.add(dirLight2);

        // Dimensions (mm scaled)
        const subW = {self.sub_w};
        const subL = {self.sub_l};
        const subYMin = {self.sub_y_min};
        const subYCenter = subYMin + subL / 2;
        const rfH = 0.4;    // Visually scaled for clear perspective
        const coreH = 1.0;  // Visually scaled FR-4 mechanical core
        const is4Layer = {str(self.is_4layer).lower()};

        if (is4Layer) {{
            // 1. Layer 4 Bottom Ground Plane
            const botGndGeo = new THREE.PlaneGeometry(subW, subL);
            const botGndMat = new THREE.MeshStandardMaterial({{
                color: 0x78350f,
                metalness: 0.85,
                roughness: 0.25,
                side: THREE.DoubleSide
            }});
            const botGndMesh = new THREE.Mesh(botGndGeo, botGndMat);
            botGndMesh.position.set(0, subYCenter, -coreH);
            scene.add(botGndMesh);

            // 2. Dielectric 2: FR-4 Core
            const coreGeo = new THREE.BoxGeometry(subW, subL, coreH);
            const coreMat = new THREE.MeshPhysicalMaterial({{
                color: 0x10b981,
                transparent: true,
                opacity: 0.35,
                roughness: 0.2,
                transmission: 0.4,
                thickness: 0.8
            }});
            const coreMesh = new THREE.Mesh(coreGeo, coreMat);
            coreMesh.position.set(0, subYCenter, -coreH / 2);
            scene.add(coreMesh);
        }}

        // 3. Layer 2 Ground Plane (z = 0)
        const gndGeo = new THREE.PlaneGeometry(subW, subL);
        const gndMat = new THREE.MeshStandardMaterial({{
            color: 0xb45309,
            metalness: 0.85,
            roughness: 0.25,
            side: THREE.DoubleSide
        }});
        const gndMesh = new THREE.Mesh(gndGeo, gndMat);
        gndMesh.position.set(0, subYCenter, 0.0);
        scene.add(gndMesh);

        // 4. Dielectric 1: Rogers RO3003
        const subGeo = new THREE.BoxGeometry(subW, subL, rfH);
        const subMat = new THREE.MeshPhysicalMaterial({{
            color: 0x93c5fd,
            transparent: true,
            opacity: 0.45,
            roughness: 0.1,
            transmission: 0.6,
            thickness: 0.5
        }});
        const subMesh = new THREE.Mesh(subGeo, subMat);
        subMesh.position.set(0, subYCenter, rfH / 2);
        scene.add(subMesh);

        // Copper generator helper
        function addTrace(xCenter, w, yStart, yEnd, colorHex) {{
            const len = yEnd - yStart;
            const yMid = yStart + len / 2;
            const geo = new THREE.BoxGeometry(w, len, 0.04);
            const mat = new THREE.MeshStandardMaterial({{
                color: colorHex,
                metalness: 0.8,
                roughness: 0.2
            }});
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(xCenter, yMid, rfH + 0.02);
            scene.add(mesh);
        }}

        // 5. Layer 1 Copper traces across all channels
        const channelOffsets = {self.channel_offsets_x};
        const patchPositions = {self.patch_positions};
        const connectPositions = {self.connect_positions};

        channelOffsets.forEach((xCenter, chIdx) => {{
            addTrace(xCenter, {self.feed_width}, {self.y_feed_start}, {self.y_feed_end}, 0xea580c);
            addTrace(xCenter, {self.trans_width}, {self.y_trans_start}, {self.y_trans_end}, 0xf59e0b);

            patchPositions.forEach((pos) => {{
                addTrace(xCenter, {self.patch_width}, pos[0], pos[1], 0x2563eb);
            }});

            connectPositions.forEach((pos) => {{
                addTrace(xCenter, {self.connect_width}, pos[0], pos[1], 0x16a34a);
            }});

            // Port marker
            const portGeo = new THREE.CylinderGeometry(0.08, 0.08, rfH, 16);
            const portMat = new THREE.MeshStandardMaterial({{ color: chIdx === 0 ? 0xdc2626 : 0x475569 }});
            const portMesh = new THREE.Mesh(portGeo, portMat);
            portMesh.rotation.x = Math.PI / 2;
            portMesh.position.set(xCenter, 0, rfH / 2);
            scene.add(portMesh);
        }});

        // Grid helper on bottom
        const gridHelper = new THREE.GridHelper(40, 40, 0x334155, 0x1e293b);
        gridHelper.rotation.x = Math.PI / 2;
        gridHelper.position.set(0, subYCenter, is4Layer ? -coreH - 0.05 : -0.05);
        scene.add(gridHelper);

        window.addEventListener('resize', onWindowResize, false);
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
