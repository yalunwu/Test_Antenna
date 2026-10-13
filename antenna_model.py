"""
Proven Antenna Models for openEMS FDTD Simulation.
Features three industry-proven Texas Instruments antenna designs:
1. TI 77 GHz RX Series-Fed Patch Array (SWRU553A Section 2.6.2, Figure 16 & Table 7)
2. TI 2.4 GHz Small Size MIFA PCB Antenna (AN043 / SWRA117D, Figure 3 & Table 1)
3. TI 2.4 GHz Inverted-F PCB Antenna (SWRU120D / DN0007, Figure 1-1 & Table 1-1)
"""
import abc
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

# ==============================================================================
# Base Antenna Model
# ==============================================================================

class BaseAntennaModel(abc.ABC):
    """Abstract base class for proven antenna simulation models."""
    def __init__(self,
                 name: str,
                 title: str,
                 doc_reference: str,
                 f_center: float,
                 f_start: float,
                 f_stop: float,
                 f_wide_start: float,
                 f_wide_stop: float,
                 cells_per_wavelength: float = 60.0):
        self.name = name
        self.title = title
        self.doc_reference = doc_reference
        self.f_center = f_center
        self.f_start = f_start
        self.f_stop = f_stop
        self.f_wide_start = f_wide_start
        self.f_wide_stop = f_wide_stop
        self.cells_per_wavelength = max(60.0, float(cells_per_wavelength))
        
        self.unit = 1e-3  # coordinate unit = 1 mm
        self.lambda0_mm = (C0 / self.f_center) / self.unit
        self.num_channels = 1
        self.channel_offsets_x = [0.0]
        self.channel_spacing = 0.0

    @abc.abstractmethod
    def create_simulation(self, end_criteria: float = 1e-4, nr_ts: int = 40000, active_channels=1):
        """Instantiate CSXCAD geometry and openEMS FDTD solver."""
        pass

    @abc.abstractmethod
    def draw_layout(self, output_path: str):
        """Generate high-resolution 2D engineering layout diagram."""
        pass

    @abc.abstractmethod
    def draw_3d_view(self, output_path: str):
        """Generate 3D isometric perspective diagram."""
        pass

    @abc.abstractmethod
    def draw_interactive_3d(self, output_path: str):
        """Generate self-contained interactive 3D WebGL HTML visualization."""
        pass

    @abc.abstractmethod
    def get_proven_specs(self) -> dict:
        """Return published benchmark specifications from the TI reference document."""
        pass

    def _smooth_mesh(self, lines: np.ndarray, max_res: float, ratio: float = 1.30) -> np.ndarray:
        """Smooth mesh lines so step size does not exceed max_res and expansion ratio."""
        lines = np.unique(np.sort(lines))
        smoothed = [lines[0]]
        for i in range(1, len(lines)):
            p0 = smoothed[-1]
            p1 = lines[i]
            dist = p1 - p0
            if dist <= 0:
                continue
            if dist > max_res:
                n_sub = int(np.ceil(dist / max_res))
                sub_pts = np.linspace(p0, p1, n_sub + 1)[1:]
                smoothed.extend(sub_pts)
            else:
                smoothed.append(p1)
        return np.array(smoothed)


# ==============================================================================
# 1. TI 77 GHz RX Antenna from SWRU553A
# ==============================================================================

class TI77GHzRXAntenna(BaseAntennaModel):
    """
    Texas Instruments 77 GHz RX Series-Fed Microstrip Patch Array Antenna.
    Direct implementation of TI Document SWRU553A (MMWCAS-RF-EVM Cascaded Radar User's Guide),
    Section 2.6.2, Figure 16 & Table 7.
    
    Stackup:
      - Layer 1 (Top Cu): 4 Chebyshev-tapered patches & GCPW-to-microstrip transition
      - Dielectric 1: 0.127 mm (5 mil) Rogers RO3003 (eps_r = 3.00, tan_d = 0.0010)
      - Layer 2: Solid RF Ground Plane at z = 0
      - Dielectric 2: 0.800 mm (31.5 mil) FR-4 Core (eps_r = 4.40, tan_d = 0.020)
      - Layer 4: Bottom Shielding Ground Plane at z = -0.80 mm
    """
    def __init__(self,
                 num_channels: int = 4,
                 channel_spacing: float = 1.950, # lambda/2 = 76.78 mils = 1.95 mm (SWRU553A Table 5)
                 cells_per_wavelength: float = 60.0,
                 is_4layer: bool = True):
        super().__init__(
            name='ti_77ghz_rx',
            title='TI 77 GHz RX Series-Fed Patch Array (SWRU553A)',
            doc_reference='Texas Instruments SWRU553A Section 2.6.2 (Table 7 & Figure 16)',
            f_center=77.0e9,
            f_start=76.0e9,
            f_stop=78.0e9,
            f_wide_start=74.0e9,
            f_wide_stop=81.0e9,
            cells_per_wavelength=cells_per_wavelength
        )
        self.num_channels = max(1, int(num_channels))
        self.channel_spacing = channel_spacing  # 1.950 mm (~lambda_0 / 2)
        self.is_4layer = is_4layer
        
        # Exact dimensions from SWRU553A Table 7 (converted from mils to mm: 1 mil = 0.0254 mm)
        self.A1_gcpw_len = 27.568 * 0.0254      # 0.7002 mm (GCPW section length)
        self.A2_trans_len = 11.802 * 0.0254     # 0.2998 mm (transition length)
        self.B1_stub_len = 47.240 * 0.0254      # 1.1999 mm (transition to Patch 1)
        self.B2_inter1_len = 49.210 * 0.0254    # 1.2499 mm (Patch 1 to Patch 2)
        self.B3_inter2_len = 45.270 * 0.0254    # 1.1500 mm (Patch 2 to Patch 3)
        self.B4_inter3_len = 49.210 * 0.0254    # 1.2499 mm (Patch 3 to Patch 4)
        
        self.C1_patch14_len = 42.520 * 0.0254   # 1.0800 mm (resonant length Patch 1 & 4)
        self.C2_patch23_len = 40.950 * 0.0254   # 1.0401 mm (resonant length Patch 2 & 3)
        
        self.D1_gcpw_w = 16.722 * 0.0254        # 0.4247 mm (GCPW signal line width)
        self.D2_trans_w = 20.002 * 0.0254       # 0.5081 mm (transition line width)
        
        self.E1_patch14_w = 39.380 * 0.0254     # 1.0003 mm (transverse width Patch 1 & 4)
        self.E2_patch23_w = 55.120 * 0.0254     # 1.4000 mm (transverse width Patch 2 & 3)
        
        self.F1_gcpw_gap = 8.400 * 0.0254       # 0.2134 mm (GCPW ground clearance gap)
        self.F2_taper_gap = 10.620 * 0.0254     # 0.2697 mm (taper ground clearance)
        self.F3_feed_w = 3.940 * 0.0254         # 0.1001 mm (interconnecting microstrip width)
        
        self.G_keepout = 78.7402 * 0.0254       # 2.0000 mm (minimum keepout margin)
        self.H_via_pitch = 20.670 * 0.0254      # 0.5250 mm (GCPW via pitch)
        
        # Substrate parameters
        self.sub_h = 0.127                      # mm (Rogers RO3003 5 mil)
        self.sub_eps_r = 3.00
        self.sub_tan_d = 0.0010
        self.sub_kappa = 2 * np.pi * self.f_center * EPS0 * self.sub_eps_r * self.sub_tan_d
        
        self.h_core = 0.800                     # mm (FR-4 core)
        self.eps_core = 4.40
        self.tan_d_core = 0.020
        self.kappa_core = 2 * np.pi * self.f_center * EPS0 * self.eps_core * self.tan_d_core
        
        # Channel offsets along x
        if self.num_channels == 1:
            self.channel_offsets_x = [0.0]
        else:
            self.channel_offsets_x = [
                (i - (self.num_channels - 1) / 2.0) * self.channel_spacing
                for i in range(self.num_channels)
            ]
            
        # Build longitudinal coordinates along y
        # y = 0 is feed entry port
        self.y_gcpw_start = 0.0
        self.y_gcpw_end = self.A1_gcpw_len
        
        self.y_trans_start = self.y_gcpw_end
        self.y_trans_end = self.y_trans_start + self.A2_trans_len
        
        self.y_stub_start = self.y_trans_end
        self.y_stub_end = self.y_stub_start + self.B1_stub_len
        
        # Patch 1
        self.y_p1_start = self.y_stub_end
        self.y_p1_end = self.y_p1_start + self.C1_patch14_len
        
        # Interconnect 1
        self.y_i1_start = self.y_p1_end
        self.y_i1_end = self.y_i1_start + self.B2_inter1_len
        
        # Patch 2
        self.y_p2_start = self.y_i1_end
        self.y_p2_end = self.y_p2_start + self.C2_patch23_len
        
        # Interconnect 2
        self.y_i2_start = self.y_p2_end
        self.y_i2_end = self.y_i2_start + self.B3_inter2_len
        
        # Patch 3
        self.y_p3_start = self.y_i2_end
        self.y_p3_end = self.y_p3_start + self.C2_patch23_len
        
        # Interconnect 3
        self.y_i3_start = self.y_p3_end
        self.y_i3_end = self.y_i3_start + self.B4_inter3_len
        
        # Patch 4
        self.y_p4_start = self.y_i3_end
        self.y_p4_end = self.y_p4_start + self.C1_patch14_len
        
        self.total_radiator_length = self.y_p4_end
        
        # Substrate boundary
        self.margin_x = 2.00
        self.margin_y = 1.50
        max_x = max(abs(x) for x in self.channel_offsets_x)
        self.sub_w = 2.0 * (max_x + self.E2_patch23_w / 2.0 + self.margin_x)
        self.sub_y_min = -self.margin_y
        self.sub_y_max = self.total_radiator_length + self.G_keepout
        self.sub_l = self.sub_y_max - self.sub_y_min

    def get_proven_specs(self) -> dict:
        return {
            "source_doc": "TI SWRU553A Section 2.6.2 (MMWCAS-RF-EVM)",
            "operating_band": "76.0 – 81.0 GHz",
            "nominal_center_freq_ghz": 77.0,
            "proven_s11_reflection_db": "< -10.0 dB (76 - 81 GHz)",
            "proven_single_element_gain_dbi": "6.9 – 7.2 dBi",
            "proven_elevation_hpbw_deg": "20.0° – 28.0° (E-Plane)",
            "proven_azimuth_hpbw_deg": "65.0° – 75.0° (H-Plane per element)",
            "proven_array_directivity_dbi": "~15.0 dBi (for 8-element RX array)",
            "substrate_material": "Rogers RO3003 (0.127 mm / 5 mil, eps_r = 3.00)",
            "board_stackup": "4-Layer mmWave PCB (RO3003 + FR-4 core)"
        }

    def create_simulation(self, end_criteria: float = 1e-4, nr_ts: int = 40000, active_channels=1):
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
        fc = 4.0e9
        FDTD.SetGaussExcite(self.f_center, fc)
        FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(self.unit)
        
        # Materials
        ro3003 = CSX.AddMaterial('RO3003', epsilon=self.sub_eps_r, kappa=self.sub_kappa)
        if self.is_4layer:
            fr4 = CSX.AddMaterial('FR4_Core', epsilon=self.eps_core, kappa=self.kappa_core)
        copper = CSX.AddMetal('Copper')
        
        # 1. Dielectric 1: Rogers RO3003 (z: 0 to sub_h)
        ro3003.AddBox(start=[-self.sub_w/2, self.sub_y_min, 0.0],
                      stop=[self.sub_w/2, self.sub_y_max, self.sub_h], priority=0)
                      
        # 2. Layer 2: RF Ground plane at z = 0
        copper.AddBox(start=[-self.sub_w/2, self.sub_y_min, 0.0],
                      stop=[self.sub_w/2, self.sub_y_max, 0.0], priority=10)
                      
        if self.is_4layer:
            # 3. Dielectric 2: FR-4 Core (z: -h_core to 0)
            fr4.AddBox(start=[-self.sub_w/2, self.sub_y_min, -self.h_core],
                       stop=[self.sub_w/2, self.sub_y_max, 0.0], priority=0)
            # 4. Layer 4: Bottom Shielding GND at z = -h_core
            copper.AddBox(start=[-self.sub_w/2, self.sub_y_min, -self.h_core],
                          stop=[self.sub_w/2, self.sub_y_max, -self.h_core], priority=10)

        # 5. Top Metallization (z = sub_h) for all channels
        gcpw_ground_w = 0.60  # coplanar ground width
        for xc in self.channel_offsets_x:
            # GCPW Feed strip
            copper.AddBox(start=[xc - self.D1_gcpw_w/2, self.y_gcpw_start, self.sub_h],
                          stop=[xc + self.D1_gcpw_w/2, self.y_gcpw_end, self.sub_h], priority=10)
                          
            # GCPW Coplanar Grounds (left and right)
            x_gnd_left_start = xc - self.D1_gcpw_w/2 - self.F1_gcpw_gap - gcpw_ground_w
            x_gnd_left_stop = xc - self.D1_gcpw_w/2 - self.F1_gcpw_gap
            copper.AddBox(start=[x_gnd_left_start, self.y_gcpw_start, self.sub_h],
                          stop=[x_gnd_left_stop, self.y_gcpw_end, self.sub_h], priority=10)
                          
            x_gnd_right_start = xc + self.D1_gcpw_w/2 + self.F1_gcpw_gap
            x_gnd_right_stop = xc + self.D1_gcpw_w/2 + self.F1_gcpw_gap + gcpw_ground_w
            copper.AddBox(start=[x_gnd_right_start, self.y_gcpw_start, self.sub_h],
                          stop=[x_gnd_right_stop, self.y_gcpw_end, self.sub_h], priority=10)

            # Transition section
            copper.AddBox(start=[xc - self.D2_trans_w/2, self.y_trans_start, self.sub_h],
                          stop=[xc + self.D2_trans_w/2, self.y_trans_end, self.sub_h], priority=10)
                          
            # Stub to Patch 1
            copper.AddBox(start=[xc - self.F3_feed_w/2, self.y_stub_start, self.sub_h],
                          stop=[xc + self.F3_feed_w/2, self.y_stub_end, self.sub_h], priority=10)
                          
            # Patch 1 (C1 x E1)
            copper.AddBox(start=[xc - self.E1_patch14_w/2, self.y_p1_start, self.sub_h],
                          stop=[xc + self.E1_patch14_w/2, self.y_p1_end, self.sub_h], priority=10)
                          
            # Interconnect 1 (B2 x F3)
            copper.AddBox(start=[xc - self.F3_feed_w/2, self.y_i1_start, self.sub_h],
                          stop=[xc + self.F3_feed_w/2, self.y_i1_end, self.sub_h], priority=10)
                          
            # Patch 2 (C2 x E2)
            copper.AddBox(start=[xc - self.E2_patch23_w/2, self.y_p2_start, self.sub_h],
                          stop=[xc + self.E2_patch23_w/2, self.y_p2_end, self.sub_h], priority=10)
                          
            # Interconnect 2 (B3 x F3)
            copper.AddBox(start=[xc - self.F3_feed_w/2, self.y_i2_start, self.sub_h],
                          stop=[xc + self.F3_feed_w/2, self.y_i2_end, self.sub_h], priority=10)
                          
            # Patch 3 (C2 x E2)
            copper.AddBox(start=[xc - self.E2_patch23_w/2, self.y_p3_start, self.sub_h],
                          stop=[xc + self.E2_patch23_w/2, self.y_p3_end, self.sub_h], priority=10)
                          
            # Interconnect 3 (B4 x F3)
            copper.AddBox(start=[xc - self.F3_feed_w/2, self.y_i3_start, self.sub_h],
                          stop=[xc + self.F3_feed_w/2, self.y_i3_end, self.sub_h], priority=10)
                          
            # Patch 4 (C1 x E1)
            copper.AddBox(start=[xc - self.E1_patch14_w/2, self.y_p4_start, self.sub_h],
                          stop=[xc + self.E1_patch14_w/2, self.y_p4_end, self.sub_h], priority=10)

        # 6. Lumped 50-Ohm Ports
        ports = []
        for ch_idx, xc in enumerate(self.channel_offsets_x):
            port_nr = ch_idx + 1
            port_start = [xc - self.D1_gcpw_w/2, 0.0, 0.0]
            port_stop = [xc + self.D1_gcpw_w/2, 0.0, self.sub_h]
            excite_val = 1.0 if port_nr in self.active_channels else 0.0
            p = FDTD.AddLumpedPort(port_nr, 50.0, port_start, port_stop, 'z', excite_val, priority=20, edges2grid='xy')
            ports.append(p)
            
        # 7. Mesh Definition
        res = self.lambda0_mm / self.cells_per_wavelength
        
        # Z lines
        nz_sub = 5
        z_sub_lines = np.linspace(0.0, self.sub_h, nz_sub + 1)
        air_pad = self.lambda0_mm * 0.50
        z_raw = np.concatenate([
            [-self.h_core - air_pad, -self.h_core] if self.is_4layer else [-air_pad],
            z_sub_lines,
            [self.sub_h + air_pad]
        ])
        z_lines = self._smooth_mesh(z_raw, res, ratio=1.30)
        
        # X lines
        x_pts = [-self.sub_w/2 - self.lambda0_mm * 0.5, -self.sub_w/2, self.sub_w/2, self.sub_w/2 + self.lambda0_mm * 0.5]
        for xc in self.channel_offsets_x:
            x_pts.extend([
                xc - self.E2_patch23_w/2, xc + self.E2_patch23_w/2,
                xc - self.E1_patch14_w/2, xc + self.E1_patch14_w/2,
                xc - self.D1_gcpw_w/2, xc + self.D1_gcpw_w/2,
                xc - self.F3_feed_w/2, xc + self.F3_feed_w/2
            ])
        x_lines = self._smooth_mesh(np.array(x_pts), res, ratio=1.35)
        
        # Y lines
        y_pts = [
            self.sub_y_min - self.lambda0_mm * 0.5, self.sub_y_min,
            0.0, self.y_gcpw_end, self.y_trans_end, self.y_stub_end,
            self.y_p1_start, self.y_p1_end, self.y_p2_start, self.y_p2_end,
            self.y_p3_start, self.y_p3_end, self.y_p4_start, self.y_p4_end,
            self.sub_y_max, self.sub_y_max + self.lambda0_mm * 0.5
        ]
        y_lines = self._smooth_mesh(np.array(y_pts), res, ratio=1.35)
        
        mesh.AddLine('x', x_lines)
        mesh.AddLine('y', y_lines)
        mesh.AddLine('z', z_lines)
        
        # 8. NF2FF Recording Box
        nf2ff = FDTD.CreateNF2FFBox()
        return FDTD, CSX, ports, nf2ff

    def draw_layout(self, output_path: str):
        fig, (ax_plan, ax_side) = plt.subplots(1, 2, figsize=(14, 7),
                                               gridspec_kw={'width_ratios': [1.3, 1.0]},
                                               tight_layout=True)
        # Plan view
        sub_rect = mpatches.Rectangle((-self.sub_w/2, self.sub_y_min), self.sub_w, self.sub_l,
                                     facecolor='#e2e8f0', edgecolor='#64748b', linewidth=1.5,
                                     label='Rogers RO3003 (Dk=3.0)')
        ax_plan.add_patch(sub_rect)
        
        for ch_idx, xc in enumerate(self.channel_offsets_x):
            # GCPW & transition
            ax_plan.add_patch(mpatches.Rectangle((xc - self.D1_gcpw_w/2, 0), self.D1_gcpw_w, self.A1_gcpw_len, facecolor='#dc2626'))
            ax_plan.add_patch(mpatches.Rectangle((xc - self.D2_trans_w/2, self.y_trans_start), self.D2_trans_w, self.A2_trans_len, facecolor='#ea580c'))
            ax_plan.add_patch(mpatches.Rectangle((xc - self.F3_feed_w/2, self.y_stub_start), self.F3_feed_w, self.B1_stub_len, facecolor='#f59e0b'))
            
            # Patches
            ax_plan.add_patch(mpatches.Rectangle((xc - self.E1_patch14_w/2, self.y_p1_start), self.E1_patch14_w, self.C1_patch14_len, facecolor='#2563eb', label='Patch 1 & 4' if ch_idx==0 else ''))
            ax_plan.add_patch(mpatches.Rectangle((xc - self.F3_feed_w/2, self.y_i1_start), self.F3_feed_w, self.B2_inter1_len, facecolor='#10b981'))
            
            ax_plan.add_patch(mpatches.Rectangle((xc - self.E2_patch23_w/2, self.y_p2_start), self.E2_patch23_w, self.C2_patch23_len, facecolor='#4f46e5', label='Patch 2 & 3' if ch_idx==0 else ''))
            ax_plan.add_patch(mpatches.Rectangle((xc - self.F3_feed_w/2, self.y_i2_start), self.F3_feed_w, self.B3_inter2_len, facecolor='#10b981'))
            
            ax_plan.add_patch(mpatches.Rectangle((xc - self.E2_patch23_w/2, self.y_p3_start), self.E2_patch23_w, self.C2_patch23_len, facecolor='#4f46e5'))
            ax_plan.add_patch(mpatches.Rectangle((xc - self.F3_feed_w/2, self.y_i3_start), self.F3_feed_w, self.B4_inter3_len, facecolor='#10b981'))
            
            ax_plan.add_patch(mpatches.Rectangle((xc - self.E1_patch14_w/2, self.y_p4_start), self.E1_patch14_w, self.C1_patch14_len, facecolor='#2563eb'))
            
            ax_plan.text(xc, -0.6, f'Ch{ch_idx+1}\nPort', ha='center', va='top', fontsize=8, fontweight='bold', color='#dc2626')
            
        ax_plan.set_xlim([-self.sub_w/2 - 0.5, self.sub_w/2 + 0.5])
        ax_plan.set_ylim([self.sub_y_min - 1.0, self.sub_y_max + 0.5])
        ax_plan.set_xlabel('X Dimension (mm)', fontsize=10)
        ax_plan.set_ylabel('Y Dimension (mm)', fontsize=10)
        ax_plan.set_title('(a) TI 77 GHz RX 4-Patch Series Array — Plan View\n(SWRU553A Fig 16: C1=1.08mm, C2=1.04mm, E1=1.00mm, E2=1.40mm)', fontsize=10.5, fontweight='bold')
        ax_plan.legend(loc='upper right', fontsize=8.5)
        ax_plan.grid(True, linestyle=':', alpha=0.6)
        
        # Side view
        if self.is_4layer:
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -self.h_core), self.sub_l, self.h_core, facecolor='#bbf7d0', edgecolor='#16a34a', label=f'FR-4 Core ({self.h_core:.2f} mm)'))
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -self.h_core - 0.02), self.sub_l, 0.02, facecolor='#b45309', label='Layer 4 Bottom GND'))
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -0.015), self.sub_l, 0.015, facecolor='#b45309', label='Layer 2 RF GND'))
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, 0), self.sub_l, self.sub_h, facecolor='#dbeafe', edgecolor='#2563eb', label=f'RO3003 Cavity ({self.sub_h:.3f} mm)'))
            ax_side.set_ylim([-self.h_core - 0.1, self.sub_h * 4.0])
        else:
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, 0), self.sub_l, self.sub_h, facecolor='#dbeafe', edgecolor='#2563eb', label='RO3003'))
            ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -0.015), self.sub_l, 0.015, facecolor='#b45309', label='GND'))
            ax_side.set_ylim([-0.05, self.sub_h * 3.5])
            
        ax_side.set_xlim([self.sub_y_min - 0.5, self.sub_y_max + 0.5])
        ax_side.set_xlabel('Y Dimension (mm)', fontsize=10)
        ax_side.set_ylabel('Z Dimension (mm)', fontsize=10)
        ax_side.set_title('(b) Automotive 4-Layer mmWave Stackup Cross-Section', fontsize=10.5, fontweight='bold')
        ax_side.legend(loc='upper right', fontsize=8.5)
        ax_side.grid(True, linestyle=':', alpha=0.6)
        
        fig.suptitle(f"{self.title}\nAuthentic Reference Geometry from {self.doc_reference}", fontsize=12, fontweight='bold')
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_3d_view(self, output_path: str):
        fig = plt.figure(figsize=(11, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        
        x0, x1 = -self.sub_w/2, self.sub_w/2
        y0, y1 = self.sub_y_min, self.sub_y_max
        z_gnd, z_rf, z_core = 0.0, 0.40, -1.00
        
        # FR-4 core
        if self.is_4layer:
            bot_face = [[[x0, y0, z_core], [x1, y0, z_core], [x1, y1, z_core], [x0, y1, z_core]]]
            ax.add_collection3d(Poly3DCollection(bot_face, facecolors='#b45309', alpha=0.9))
            core_faces = [
                [[x0, y0, z_core], [x1, y0, z_core], [x1, y1, z_core], [x0, y1, z_core]],
                [[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y1, z_gnd], [x0, y1, z_gnd]],
                [[x0, y0, z_core], [x1, y0, z_core], [x1, y0, z_gnd], [x0, y0, z_gnd]],
                [[x0, y1, z_core], [x1, y1, z_core], [x1, y1, z_gnd], [x0, y1, z_gnd]]
            ]
            ax.add_collection3d(Poly3DCollection(core_faces, facecolors='#86efac', edgecolors='#22c55e', alpha=0.2))
            
        # Ground plane
        gnd_face = [[[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y1, z_gnd], [x0, y1, z_gnd]]]
        ax.add_collection3d(Poly3DCollection(gnd_face, facecolors='#b45309', alpha=0.85))
        
        # Substrate
        sub_faces = [
            [[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y1, z_gnd], [x0, y1, z_gnd]],
            [[x0, y0, z_rf], [x1, y0, z_rf], [x1, y1, z_rf], [x0, y1, z_rf]],
            [[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y0, z_rf], [x0, y0, z_rf]],
            [[x0, y1, z_gnd], [x1, y1, z_gnd], [x1, y1, z_rf], [x0, y1, z_rf]]
        ]
        ax.add_collection3d(Poly3DCollection(sub_faces, facecolors='#dbeafe', edgecolors='#93c5fd', alpha=0.35))
        
        def add_strip(xc, xw, ys, ye, color):
            z_cu = z_rf + 0.04
            p = [[[xc - xw/2, ys, z_cu], [xc + xw/2, ys, z_cu], [xc + xw/2, ye, z_cu], [xc - xw/2, ye, z_cu]]]
            ax.add_collection3d(Poly3DCollection(p, facecolors=color, edgecolors='#78350f', alpha=1.0, linewidth=0.8))
            
        for ch_idx, xc in enumerate(self.channel_offsets_x):
            add_strip(xc, self.D1_gcpw_w, 0, self.A1_gcpw_len, '#dc2626')
            add_strip(xc, self.D2_trans_w, self.y_trans_start, self.y_trans_end, '#ea580c')
            add_strip(xc, self.F3_feed_w, self.y_stub_start, self.y_stub_end, '#f59e0b')
            add_strip(xc, self.E1_patch14_w, self.y_p1_start, self.y_p1_end, '#2563eb')
            add_strip(xc, self.F3_feed_w, self.y_i1_start, self.y_i1_end, '#10b981')
            add_strip(xc, self.E2_patch23_w, self.y_p2_start, self.y_p2_end, '#4f46e5')
            add_strip(xc, self.F3_feed_w, self.y_i2_start, self.y_i2_end, '#10b981')
            add_strip(xc, self.E2_patch23_w, self.y_p3_start, self.y_p3_end, '#4f46e5')
            add_strip(xc, self.F3_feed_w, self.y_i3_start, self.y_i3_end, '#10b981')
            add_strip(xc, self.E1_patch14_w, self.y_p4_start, self.y_p4_end, '#2563eb')
            
            ax.plot([xc, xc], [0, 0], [z_gnd, z_rf], color='#dc2626', linewidth=3)
            ax.text(xc, -0.6, z_rf/2, f'P{ch_idx+1}', color='#dc2626', fontweight='bold', fontsize=8)

        ax.set_xlim([-self.sub_w/2 - 0.5, self.sub_w/2 + 0.5])
        ax.set_ylim([self.sub_y_min - 0.5, self.sub_y_max + 0.5])
        ax.set_zlim([z_core - 0.2 if self.is_4layer else -0.2, z_rf + 0.8])
        ax.view_init(elev=28, azim=-55)
        ax.set_title(f"{self.title}\n3D Isometric Structural Model", fontsize=11, fontweight='bold')
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_interactive_3d(self, output_path: str):
        specs = self.get_proven_specs()
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.title} — 3D Interactive Model</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc; }}
        #hud {{ position: absolute; top: 16px; left: 16px; background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px); padding: 16px 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.12); max-width: 340px; font-size: 12px; }}
        h2 {{ margin: 0 0 8px 0; font-size: 15px; color: #38bdf8; font-weight: 700; }}
        .metric {{ font-weight: 600; color: #f1f5f9; }}
        .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; background: #0284c7; color: white; margin-bottom: 8px; }}
        .proven {{ border-top: 1px solid rgba(255,255,255,0.1); margin-top: 10px; padding-top: 8px; font-size: 11px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="canvas-container"></div>
    <div id="hud">
        <span class="badge">PROVEN REFERENCE MODEL</span>
        <h2>{self.title}</h2>
        <p>Reference: <span class="metric">{self.doc_reference}</span></p>
        <p>Operating Band: <span class="metric">{specs['operating_band']}</span></p>
        <p>Channels: <span class="metric">{self.num_channels} RX Channels (d = {self.channel_spacing:.2f} mm)</span></p>
        <p>Series Elements: <span class="metric">4 Chebyshev-Tapered Patches</span></p>
        <p>Stackup: <span class="metric">{specs['substrate_material']}</span></p>
        <div class="proven">
            <strong style="color: #38bdf8;">TI Published Proven Metrics:</strong><br>
            • Reflection S11: {specs['proven_s11_reflection_db']}<br>
            • Peak Gain: {specs['proven_single_element_gain_dbi']}<br>
            • Elevation HPBW: {specs['proven_elevation_hpbw_deg']}<br>
            • Azimuth HPBW: {specs['proven_azimuth_hpbw_deg']}
        </div>
    </div>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
        camera.position.set(12, -18, 16);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, {self.total_radiator_length/2:.1f}, 0);
        controls.update();

        const light1 = new THREE.DirectionalLight(0xffffff, 1.0); light1.position.set(10, 10, 20); scene.add(light1);
        const light2 = new THREE.DirectionalLight(0x93c5fd, 0.5); light2.position.set(-10, -10, -10); scene.add(light2);
        scene.add(new THREE.AmbientLight(0x475569));

        // Substrate
        const subGeo = new THREE.BoxGeometry({self.sub_w:.2f}, {self.sub_l:.2f}, 0.2);
        const subMat = new THREE.MeshStandardMaterial({{ color: 0x3b82f6, transparent: true, opacity: 0.35 }});
        const sub = new THREE.Mesh(subGeo, subMat);
        sub.position.set(0, {self.sub_y_min + self.sub_l/2:.2f}, 0.1);
        scene.add(sub);

        // Ground plane
        const gndGeo = new THREE.BoxGeometry({self.sub_w:.2f}, {self.sub_l:.2f}, 0.04);
        const gndMat = new THREE.MeshStandardMaterial({{ color: 0xb45309, metalness: 0.8, roughness: 0.3 }});
        const gnd = new THREE.Mesh(gndGeo, gndMat);
        gnd.position.set(0, {self.sub_y_min + self.sub_l/2:.2f}, -0.02);
        scene.add(gnd);

        // Patches and traces
        const cuMat = new THREE.MeshStandardMaterial({{ color: 0xf59e0b, metalness: 0.9, roughness: 0.2 }});
        const patchMat = new THREE.MeshStandardMaterial({{ color: 0x2563eb, metalness: 0.7, roughness: 0.3 }});

        function addBox(x, y, w, l, mat) {{
            const geo = new THREE.BoxGeometry(w, l, 0.05);
            const m = new THREE.Mesh(geo, mat);
            m.position.set(x, y + l/2, 0.22);
            scene.add(m);
        }}

        const offsets = {self.channel_offsets_x};
        offsets.forEach(xc => {{
            addBox(xc, 0, {self.D1_gcpw_w:.3f}, {self.A1_gcpw_len:.3f}, cuMat);
            addBox(xc, {self.y_trans_start:.3f}, {self.D2_trans_w:.3f}, {self.A2_trans_len:.3f}, cuMat);
            addBox(xc, {self.y_stub_start:.3f}, {self.F3_feed_w:.3f}, {self.B1_stub_len:.3f}, cuMat);
            addBox(xc, {self.y_p1_start:.3f}, {self.E1_patch14_w:.3f}, {self.C1_patch14_len:.3f}, patchMat);
            addBox(xc, {self.y_i1_start:.3f}, {self.F3_feed_w:.3f}, {self.B2_inter1_len:.3f}, cuMat);
            addBox(xc, {self.y_p2_start:.3f}, {self.E2_patch23_w:.3f}, {self.C2_patch23_len:.3f}, patchMat);
            addBox(xc, {self.y_i2_start:.3f}, {self.F3_feed_w:.3f}, {self.B3_inter2_len:.3f}, cuMat);
            addBox(xc, {self.y_p3_start:.3f}, {self.E2_patch23_w:.3f}, {self.C2_patch23_len:.3f}, patchMat);
            addBox(xc, {self.y_i3_start:.3f}, {self.F3_feed_w:.3f}, {self.B4_inter3_len:.3f}, cuMat);
            addBox(xc, {self.y_p4_start:.3f}, {self.E1_patch14_w:.3f}, {self.C1_patch14_len:.3f}, patchMat);
        }});

        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
        animate();
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)


# ==============================================================================
# 2. TI 2.4 GHz Small Size MIFA from AN043 (SWRA117D)
# ==============================================================================

class TI24GHzAN043Antenna(BaseAntennaModel):
    """
    Texas Instruments Small Size 2.4 GHz PCB Meandered Inverted-F Antenna (MIFA).
    Direct implementation of TI Application Note AN043 / SWRA117D (Figure 3 & Table 1).
    Used in CC2511 USB Dongle Reference Design.
    
    Footprint: 15.2 mm x 5.7 mm (compact meandered IFA).
    Stackup:
      - Layer 1 (Top Cu): Meandered IFA radiator with 50-Ohm feed and shorting pin
      - Dielectric: FR-4, eps_r = 4.5, tan_d = 0.02, thickness = 1.00 mm
      - Ground: Continuous RF Ground Plane on bottom / internal layer
    """
    def __init__(self,
                 cells_per_wavelength: float = 60.0,
                 sub_h: float = 1.00,
                 ground_length: float = 30.0):
        super().__init__(
            name='ti_24ghz_an043',
            title='TI 2.4 GHz Small Size MIFA (AN043 / SWRA117D)',
            doc_reference='Texas Instruments AN043 / SWRA117D (Table 1 & Figure 3)',
            f_center=2.44e9,
            f_start=2.40e9,
            f_stop=2.4835e9,
            f_wide_start=2.20e9,
            f_wide_stop=2.70e9,
            cells_per_wavelength=cells_per_wavelength
        )
        self.sub_h = sub_h
        self.sub_eps_r = 4.50
        self.sub_tan_d = 0.020
        self.sub_kappa = 2 * np.pi * self.f_center * EPS0 * self.sub_eps_r * self.sub_tan_d
        self.ground_length = ground_length
        
        # Dimensions from AN043 Table 1 (mm)
        self.L1 = 3.94
        self.L2 = 2.70
        self.L3 = 5.00
        self.L4 = 2.64
        self.L5 = 2.00
        self.L6 = 4.90
        self.W1 = 0.90
        self.W2 = 0.50
        self.D1 = 0.50
        self.D2 = 0.30
        self.D3 = 0.30
        self.D4 = 0.50
        self.D5 = 1.40
        self.D6 = 1.70
        
        # Board dimensions
        self.antenna_footprint_w = 15.20
        self.antenna_footprint_h = 5.70
        self.sub_w = 18.00                     # board width with margins
        self.board_x_min = -self.sub_w / 2.0
        self.board_x_max = self.sub_w / 2.0
        self.sub_y_min = -self.ground_length   # ground plane extends into -y
        self.sub_y_max = self.antenna_footprint_h + 1.50
        self.sub_l = self.sub_y_max - self.sub_y_min

    def get_proven_specs(self) -> dict:
        return {
            "source_doc": "TI AN043 / SWRA117D (CC2511 USB Dongle Reference Design)",
            "operating_band": "2.400 – 2.4835 GHz ISM Band",
            "nominal_center_freq_ghz": 2.44,
            "proven_s11_reflection_db": "< -10.0 dB across ISM band (typ. -15 to -20 dB)",
            "proven_10db_bandwidth_mhz": "250 MHz (free space), 100 MHz (laptop attached)",
            "proven_peak_gain_xy_dbi": "4.5 dB (free space), 3.3 dB (with laptop)",
            "proven_peak_gain_xz_dbi": "5.3 dB",
            "proven_peak_gain_yz_dbi": "5.3 dB",
            "proven_los_range_meters": "240 m (250 kbps, 1% PER)",
            "antenna_dimensions": "15.2 mm x 5.7 mm compact footprint",
            "substrate_material": "Standard FR-4 (eps_r = 4.5, tan_d = 0.02, 1.0 mm thickness)"
        }

    def _get_trace_boxes(self):
        """Compute coordinates of all copper trace segments for AN043 MIFA."""
        # Align left edge of antenna footprint to x_origin
        x0 = -self.antenna_footprint_w / 2.0 + self.D1
        boxes = []
        
        # 1. Left shorting vertical trace (connected to GND at y=0)
        # Runs from y=0 to y=L6 with width W1 at bottom and W2 upper
        boxes.append(('gnd_strap', x0, x0 + self.W1, 0.0, self.D4))
        boxes.append(('left_stem', x0, x0 + self.W2, self.D4, self.L6))
        
        # 2. Feed vertical trace (from y=0 to y=L6)
        x_feed_left = x0 + self.W1 + self.D5
        x_feed_right = x_feed_left + self.W2
        boxes.append(('feed_line', x_feed_left, x_feed_right, 0.0, self.L6))
        
        # 3. Top horizontal bar over left section (length L3 = 5.0 mm)
        boxes.append(('top_bar_1', x0, x0 + self.L3, self.L6, self.L6 + self.W2))
        
        # 4. First downward meander tooth (length L4 = 2.64 mm)
        x_t1_r = x0 + self.L3
        x_t1_l = x_t1_r - self.W2
        y_bot_1 = self.L6 - self.L4
        boxes.append(('tooth_down_1', x_t1_l, x_t1_r, y_bot_1, self.L6))
        
        # 5. First bottom horizontal link (inner length L5 = 2.0 mm)
        x_b1_r = x_t1_r + self.L5
        boxes.append(('bot_link_1', x_t1_l, x_b1_r, y_bot_1, y_bot_1 + self.W2))
        
        # 6. First upward tooth (length L4 = 2.64 mm)
        x_u1_l = x_b1_r - self.W2
        x_u1_r = x_b1_r
        boxes.append(('tooth_up_1', x_u1_l, x_u1_r, y_bot_1, self.L6))
        
        # 7. Second top horizontal link (inner length L2 = 2.7 mm)
        x_top2_r = x_u1_r + self.L2
        boxes.append(('top_link_2', x_u1_l, x_top2_r, self.L6, self.L6 + self.W2))
        
        # 8. Second downward tooth (length L4 = 2.64 mm)
        x_t2_r = x_top2_r
        x_t2_l = x_t2_r - self.W2
        boxes.append(('tooth_down_2', x_t2_l, x_t2_r, y_bot_1, self.L6))
        
        # 9. Second bottom horizontal link (inner length L5 = 2.0 mm)
        x_b2_r = x_t2_r + self.L5
        boxes.append(('bot_link_2', x_t2_l, x_b2_r, y_bot_1, y_bot_1 + self.W2))
        
        # 10. Second upward tooth (length L4 = 2.64 mm)
        x_u2_l = x_b2_r - self.W2
        x_u2_r = x_b2_r
        boxes.append(('tooth_up_2', x_u2_l, x_u2_r, y_bot_1, self.L6))
        
        # 11. Third top horizontal link (inner length L2 = 2.7 mm)
        x_top3_r = x_u2_r + self.L2
        boxes.append(('top_link_3', x_u2_l, x_top3_r, self.L6, self.L6 + self.W2))
        
        # 12. Final downward tail (length L1 = 3.94 mm)
        x_tail_r = x_top3_r
        x_tail_l = x_tail_r - self.W2
        y_tail_bot = self.L6 + self.W2 - self.L1
        boxes.append(('tail_down', x_tail_l, x_tail_r, y_tail_bot, self.L6 + self.W2))
        
        return boxes, (x_feed_left, x_feed_right)

    def create_simulation(self, end_criteria: float = 1e-4, nr_ts: int = 35000, active_channels=1):
        FDTD = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
        fc = 1.0e9  # 1 GHz bandwidth around 2.44 GHz
        FDTD.SetGaussExcite(self.f_center, fc)
        FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(self.unit)
        
        # Materials
        fr4 = CSX.AddMaterial('FR4', epsilon=self.sub_eps_r, kappa=self.sub_kappa)
        copper = CSX.AddMetal('Copper')
        
        # Substrate block
        fr4.AddBox(start=[self.board_x_min, self.sub_y_min, 0.0],
                   stop=[self.board_x_max, self.sub_y_max, self.sub_h], priority=0)
                   
        # Ground plane at z = 0 for y <= 0
        copper.AddBox(start=[self.board_x_min, self.sub_y_min, 0.0],
                      stop=[self.board_x_max, 0.0, 0.0], priority=10)
                      
        # Antenna traces on top copper (z = sub_h)
        boxes, (x_feed_l, x_feed_r) = self._get_trace_boxes()
        for name, xl, xr, yb, yt in boxes:
            copper.AddBox(start=[xl, yb, self.sub_h], stop=[xr, yt, self.sub_h], priority=10)
            
        # Ground strap via to bottom ground (at leftmost via pad)
        x0 = -self.antenna_footprint_w / 2.0 + self.D1
        copper.AddBox(start=[x0, 0.0, 0.0], stop=[x0 + self.W1, self.D4, self.sub_h], priority=10)
        
        # 50-Ohm Port across z at feed entry (y = 0)
        port_start = [x_feed_l, 0.0, 0.0]
        port_stop = [x_feed_r, 0.0, self.sub_h]
        port = FDTD.AddLumpedPort(1, 50.0, port_start, port_stop, 'z', 1.0, priority=20, edges2grid='xy')
        
        # Mesh generation (guaranteeing at least 60 cells/lambda in all dimensions)
        res = self.lambda0_mm / self.cells_per_wavelength
        air_pad = self.lambda0_mm * 0.45                  # ~55 mm air buffer before PML
        
        # Z lines
        nz_sub = 4
        z_sub = np.linspace(0.0, self.sub_h, nz_sub + 1)
        z_raw = np.concatenate([[-air_pad], z_sub, [self.sub_h + air_pad]])
        z_lines = self._smooth_mesh(z_raw, res, ratio=1.30)
        
        # X lines with air buffer
        x_features = [self.board_x_min - air_pad, self.board_x_min, self.board_x_max, self.board_x_max + air_pad]
        for _, xl, xr, _, _ in boxes:
            x_features.extend([xl, xr])
        x_lines = self._smooth_mesh(np.array(x_features), res, ratio=1.30)
        
        # Y lines with air buffer
        y_features = [self.sub_y_min - air_pad, self.sub_y_min, 0.0, self.sub_y_max, self.sub_y_max + air_pad]
        for _, _, _, yb, yt in boxes:
            y_features.extend([yb, yt])
        y_lines = self._smooth_mesh(np.array(y_features), res, ratio=1.30)
        
        mesh.AddLine('x', x_lines)
        mesh.AddLine('y', y_lines)
        mesh.AddLine('z', z_lines)
        
        nf2ff = FDTD.CreateNF2FFBox()
        return FDTD, CSX, [port], nf2ff

    def draw_layout(self, output_path: str):
        fig, (ax_plan, ax_side) = plt.subplots(1, 2, figsize=(14, 7),
                                               gridspec_kw={'width_ratios': [1.3, 1.0]},
                                               tight_layout=True)
        # Substrate
        ax_plan.add_patch(mpatches.Rectangle((self.board_x_min, self.sub_y_min), self.sub_w, self.sub_l,
                                            facecolor='#f1f5f9', edgecolor='#94a3b8', linewidth=1.2, label='FR-4 Substrate'))
        # Ground plane
        ax_plan.add_patch(mpatches.Rectangle((self.board_x_min, self.sub_y_min), self.sub_w, self.ground_length,
                                            facecolor='#cbd5e1', edgecolor='#64748b', linewidth=1.0, label='Ground Plane (Layer 2)'))
        ax_plan.text(0, -self.ground_length/2, f'Ground Plane ({self.sub_w:.0f} x {self.ground_length:.0f} mm)',
                     ha='center', va='center', color='#475569', fontweight='bold', fontsize=11)
                     
        # Antenna traces
        boxes, (x_fl, x_fr) = self._get_trace_boxes()
        for name, xl, xr, yb, yt in boxes:
            color = '#dc2626' if 'feed' in name else ('#b45309' if 'gnd' in name else '#2563eb')
            ax_plan.add_patch(mpatches.Rectangle((xl, yb), xr - xl, yt - yb, facecolor=color, edgecolor='#1e293b', linewidth=0.8))
            
        # Annotations
        ax_plan.annotate('50 $\\Omega$ Feed Point', xy=((x_fl+x_fr)/2, 0), xytext=((x_fl+x_fr)/2 + 2.0, -4.0),
                         arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5), color='#dc2626', fontweight='bold', fontsize=9)
        ax_plan.annotate('Via / Strap to GND', xy=(-self.antenna_footprint_w/2 + self.D1 + self.W1/2, 0.2), xytext=(-self.antenna_footprint_w/2 - 2.5, -4.0),
                         arrowprops=dict(arrowstyle='->', color='#b45309', lw=1.5), color='#b45309', fontweight='bold', fontsize=9)
                         
        ax_plan.set_xlim([self.board_x_min - 1.0, self.board_x_max + 1.0])
        ax_plan.set_ylim([self.sub_y_min - 2.0, self.sub_y_max + 1.0])
        ax_plan.set_xlabel('X Dimension (mm)', fontsize=10)
        ax_plan.set_ylabel('Y Dimension (mm)', fontsize=10)
        ax_plan.set_title('(a) TI AN043 2.4 GHz MIFA — Plan Layout\n(Footprint: 15.2 x 5.7 mm, L3=5.0mm, L2=2.7mm, L6=4.9mm, W2=0.5mm)', fontsize=10.5, fontweight='bold')
        ax_plan.grid(True, linestyle=':', alpha=0.6)
        
        # Side view
        ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -0.02), self.sub_l, 0.02, facecolor='#b45309', label='Bottom GND'))
        ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, 0), self.sub_l, self.sub_h, facecolor='#dbeafe', edgecolor='#3b82f6', label=f'FR-4 Substrate ({self.sub_h:.2f} mm)'))
        ax_side.add_patch(mpatches.Rectangle((0, self.sub_h), self.antenna_footprint_h, 0.035, facecolor='#2563eb', label='Top Cu Radiator'))
        ax_side.set_xlim([self.sub_y_min - 2.0, self.sub_y_max + 2.0])
        ax_side.set_ylim([-0.1, self.sub_h * 3.0])
        ax_side.set_xlabel('Y Dimension (mm)', fontsize=10)
        ax_side.set_ylabel('Z Dimension (mm)', fontsize=10)
        ax_side.set_title('(b) PCB Stackup Cross-Section', fontsize=10.5, fontweight='bold')
        ax_side.legend(loc='upper right', fontsize=8.5)
        ax_side.grid(True, linestyle=':', alpha=0.6)
        
        fig.suptitle(f"{self.title}\n{self.doc_reference}", fontsize=12, fontweight='bold')
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_3d_view(self, output_path: str):
        fig = plt.figure(figsize=(11, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        
        x0, x1 = self.board_x_min, self.board_x_max
        y0, y1 = self.sub_y_min, self.sub_y_max
        z_gnd, z_top = 0.0, self.sub_h
        
        # Ground plane at bottom
        gnd_face = [[[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, 0, z_gnd], [x0, 0, z_gnd]]]
        ax.add_collection3d(Poly3DCollection(gnd_face, facecolors='#b45309', alpha=0.85))
        
        # Substrate
        sub_faces = [
            [[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y1, z_gnd], [x0, y1, z_gnd]],
            [[x0, y0, z_top], [x1, y0, z_top], [x1, y1, z_top], [x0, y1, z_top]]
        ]
        ax.add_collection3d(Poly3DCollection(sub_faces, facecolors='#dbeafe', edgecolors='#93c5fd', alpha=0.3))
        
        # Top traces
        boxes, (x_fl, x_fr) = self._get_trace_boxes()
        for name, xl, xr, yb, yt in boxes:
            p = [[[xl, yb, z_top], [xr, yb, z_top], [xr, yt, z_top], [xl, yt, z_top]]]
            color = '#dc2626' if 'feed' in name else '#2563eb'
            ax.add_collection3d(Poly3DCollection(p, facecolors=color, alpha=1.0))
            
        ax.plot([(x_fl+x_fr)/2, (x_fl+x_fr)/2], [0, 0], [z_gnd, z_top], color='#dc2626', linewidth=3)
        ax.set_xlim([x0 - 2, x1 + 2])
        ax.set_ylim([y0 - 2, y1 + 2])
        ax.set_zlim([-0.5, z_top + 2.0])
        ax.view_init(elev=32, azim=-60)
        ax.set_title(f"{self.title}\n3D Isometric Model", fontsize=11, fontweight='bold')
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_interactive_3d(self, output_path: str):
        specs = self.get_proven_specs()
        boxes, (x_fl, x_fr) = self._get_trace_boxes()
        boxes_js = []
        for name, xl, xr, yb, yt in boxes:
            boxes_js.append(f"{{ xl: {xl:.2f}, xr: {xr:.2f}, yb: {yb:.2f}, yt: {yt:.2f}, isFeed: {'true' if 'feed' in name else 'false'} }}")
        boxes_str = ",\n            ".join(boxes_js)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.title} — 3D Interactive Model</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc; }}
        #hud {{ position: absolute; top: 16px; left: 16px; background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px); padding: 16px 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.12); max-width: 340px; font-size: 12px; }}
        h2 {{ margin: 0 0 8px 0; font-size: 15px; color: #38bdf8; font-weight: 700; }}
        .metric {{ font-weight: 600; color: #f1f5f9; }}
        .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; background: #0284c7; color: white; margin-bottom: 8px; }}
        .proven {{ border-top: 1px solid rgba(255,255,255,0.1); margin-top: 10px; padding-top: 8px; font-size: 11px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <span class="badge">PROVEN REFERENCE MODEL</span>
        <h2>{self.title}</h2>
        <p>Reference: <span class="metric">{self.doc_reference}</span></p>
        <p>Operating Band: <span class="metric">{specs['operating_band']}</span></p>
        <p>Footprint: <span class="metric">{specs['antenna_dimensions']}</span></p>
        <p>Substrate: <span class="metric">{specs['substrate_material']}</span></p>
        <div class="proven">
            <strong style="color: #38bdf8;">TI Published Proven Metrics:</strong><br>
            • Reflection S11: {specs['proven_s11_reflection_db']}<br>
            • -10 dB Bandwidth: {specs['proven_10db_bandwidth_mhz']}<br>
            • Gain XY Plane: {specs['proven_peak_gain_xy_dbi']}<br>
            • Gain XZ / YZ: {specs['proven_peak_gain_xz_dbi']}<br>
            • LOS Range: {specs['proven_los_range_meters']}
        </div>
    </div>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
        camera.position.set(20, -25, 25);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0, 0);
        controls.update();

        const light1 = new THREE.DirectionalLight(0xffffff, 1.0); light1.position.set(15, 20, 25); scene.add(light1);
        const light2 = new THREE.DirectionalLight(0x93c5fd, 0.6); light2.position.set(-15, -20, -10); scene.add(light2);
        scene.add(new THREE.AmbientLight(0x475569));

        // Substrate
        const subGeo = new THREE.BoxGeometry({self.sub_w:.2f}, {self.sub_l:.2f}, {self.sub_h:.2f});
        const subMat = new THREE.MeshStandardMaterial({{ color: 0x3b82f6, transparent: true, opacity: 0.35 }});
        const sub = new THREE.Mesh(subGeo, subMat);
        sub.position.set(0, {self.sub_y_min + self.sub_l/2:.2f}, {self.sub_h/2:.2f});
        scene.add(sub);

        // Ground Plane (bottom)
        const gndGeo = new THREE.BoxGeometry({self.sub_w:.2f}, {self.ground_length:.2f}, 0.05);
        const gndMat = new THREE.MeshStandardMaterial({{ color: 0xb45309, metalness: 0.8, roughness: 0.3 }});
        const gnd = new THREE.Mesh(gndGeo, gndMat);
        gnd.position.set(0, {-self.ground_length/2:.2f}, -0.02);
        scene.add(gnd);

        // Antenna traces
        const traceMat = new THREE.MeshStandardMaterial({{ color: 0x2563eb, metalness: 0.7, roughness: 0.3 }});
        const feedMat = new THREE.MeshStandardMaterial({{ color: 0xdc2626, metalness: 0.8, roughness: 0.2 }});
        const traces = [
            {boxes_str}
        ];
        traces.forEach(t => {{
            const w = t.xr - t.xl;
            const l = t.yt - t.yb;
            const geo = new THREE.BoxGeometry(w, l, 0.06);
            const m = new THREE.Mesh(geo, t.isFeed ? feedMat : traceMat);
            m.position.set(t.xl + w/2, t.yb + l/2, {self.sub_h:.2f} + 0.03);
            scene.add(m);
        }});

        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
        animate();
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)


# ==============================================================================
# 3. TI 2.4 GHz Inverted-F Antenna from SWRU120D (DN0007)
# ==============================================================================

class TI24GHzSWRU120DAntenna(BaseAntennaModel):
    """
    Texas Instruments 2.4 GHz Inverted-F PCB Antenna (IFA).
    Direct implementation of TI Design Note DN0007 / SWRU120D (Figure 1-1 & Table 1-1).
    Used in CC2430DB, CC2530, and CC2511 reference designs.
    
    Footprint: 25.7 mm x 7.5 mm.
    Stackup:
      - Layer 1 (Top Cu): Inverted-F radiator with tuned feed & ground loop
      - Dielectric: FR-4, eps_r = 4.40, tan_d = 0.020, thickness = 1.00 mm
      - Ground: Coplanar/bottom ground plane with feed gap
    """
    def __init__(self,
                 cells_per_wavelength: float = 60.0,
                 sub_h: float = 1.00,
                 ground_length: float = 30.0):
        super().__init__(
            name='ti_24ghz_swru120d',
            title='TI 2.4 GHz Inverted-F Antenna (SWRU120D / DN0007)',
            doc_reference='Texas Instruments SWRU120D / DN0007 (Table 1-1 & Figure 1-1)',
            f_center=2.44e9,
            f_start=2.40e9,
            f_stop=2.4835e9,
            f_wide_start=2.20e9,
            f_wide_stop=2.70e9,
            cells_per_wavelength=cells_per_wavelength
        )
        self.sub_h = sub_h
        self.sub_eps_r = 4.40
        self.sub_tan_d = 0.020
        self.sub_kappa = 2 * np.pi * self.f_center * EPS0 * self.sub_eps_r * self.sub_tan_d
        self.ground_length = ground_length
        
        # Exact dimensions from SWRU120D Table 1-1 (mm)
        self.H1 = 5.70
        self.H2 = 0.74
        self.H3 = 1.29
        self.H4 = 2.21
        self.H5 = 0.66
        self.H6 = 1.21
        self.H7 = 0.80
        self.H8 = 1.80
        self.H9 = 0.61
        self.W1 = 1.21
        self.W2 = 0.46
        self.L1 = 25.58
        self.L2 = 16.40
        self.L3 = 2.18
        self.L4 = 4.80
        self.L5 = 1.00
        self.L6 = 1.00
        self.L7 = 3.20
        self.L8 = 0.45
        
        # Board dimensions
        self.sub_w = 32.00
        self.sub_y_min = -self.ground_length
        self.sub_y_max = self.H1 + self.W1 + self.H9 + 1.00
        self.sub_l = self.sub_y_max - self.sub_y_min

    def get_proven_specs(self) -> dict:
        return {
            "source_doc": "TI SWRU120D / DN0007 (CC2430DB / CC2530 Reference Design)",
            "operating_band": "2.400 – 2.4835 GHz ISM Band",
            "nominal_center_freq_ghz": 2.44,
            "proven_s11_reflection_db": "< -15.0 dB at 2.44 GHz",
            "proven_10db_bandwidth_mhz": "> 300 MHz",
            "proven_gain_xy_plane_dbi": "1.1 dBi",
            "proven_gain_xz_plane_dbi": "3.3 dBi",
            "proven_gain_yz_plane_dbi": "1.6 dBi",
            "radiation_pattern": "Omnidirectional in PCB plane (two narrow nulls)",
            "antenna_dimensions": "25.7 mm x 7.5 mm footprint",
            "substrate_material": "Standard FR-4 (eps_r = 4.4, 1.0 mm thickness)"
        }

    def _get_trace_boxes(self):
        """Compute coordinates of all copper trace segments for SWRU120D IFA."""
        # Align origin x=0 with feed line
        x_feed = 0.0
        x_feed_l = x_feed - self.W2 / 2.0
        x_feed_r = x_feed + self.W2 / 2.0
        
        # The top horizontal radiator runs from x_left to x_right
        # Total length L1 = 25.58 mm, open arm extends L2 = 16.40 mm past feed
        x_top_r = x_feed_r + self.L2
        x_top_l = x_top_r - self.L1
        
        y_top_bot = self.H1
        y_top_top = self.H1 + self.W1
        
        boxes = []
        # 1. Main top radiator arm
        boxes.append(('top_arm', x_top_l, x_top_r, y_top_bot, y_top_top))
        
        # 2. Left vertical arm connecting top arm to bottom loop
        x_vert_w = (self.L1 - self.L2) - self.L4 - (self.L5 + self.L6 + self.L7)
        x_vert_w = max(x_vert_w, 1.20)
        boxes.append(('left_vert_arm', x_top_l, x_top_l + x_vert_w, self.H2, y_top_bot))
        
        # 3. Bottom horizontal arm
        boxes.append(('bot_arm', x_top_l, x_feed_l - self.L8, self.H2, self.H2 + self.H3))
        
        # 4. Ground shorting block to ground plane (y: 0 to H2)
        boxes.append(('gnd_short', x_top_l + self.L3, x_feed_l - self.L8, 0.0, self.H2))
        
        # 5. Feed trace entering at 50-ohm feed point
        boxes.append(('feed_trace_vert', x_feed_l, x_feed_r, 0.0, self.H5 + self.H6))
        boxes.append(('feed_trace_upper', x_feed_l - self.L7, x_feed_r, self.H5 + self.H6, self.H5 + self.H6 + self.H7))
        boxes.append(('feed_trace_neck', x_feed_l - self.L7, x_feed_l - self.L7 + self.L5, self.H5 + self.H6 + self.H7, y_top_bot))
        
        return boxes, (x_feed_l, x_feed_r), (x_top_l, x_top_r)

    def create_simulation(self, end_criteria: float = 1e-4, nr_ts: int = 35000, active_channels=1):
        FDTD = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
        fc = 1.0e9
        FDTD.SetGaussExcite(self.f_center, fc)
        FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(self.unit)
        
        # Materials
        fr4 = CSX.AddMaterial('FR4', epsilon=self.sub_eps_r, kappa=self.sub_kappa)
        copper = CSX.AddMetal('Copper')
        
        # Substrate
        fr4.AddBox(start=[-self.sub_w/2, self.sub_y_min, 0.0],
                   stop=[self.sub_w/2, self.sub_y_max, self.sub_h], priority=0)
                   
        # Ground plane at z = 0 (bottom) with slot for feed
        boxes, (x_fl, x_fr), (xt_l, xt_r) = self._get_trace_boxes()
        copper.AddBox(start=[-self.sub_w/2, self.sub_y_min, 0.0],
                      stop=[x_fl - self.L8, 0.0, 0.0], priority=10)
        copper.AddBox(start=[x_fr + self.L8, self.sub_y_min, 0.0],
                      stop=[self.sub_w/2, 0.0, 0.0], priority=10)
                      
        # Antenna traces on top copper (z = sub_h)
        for name, xl, xr, yb, yt in boxes:
            copper.AddBox(start=[xl, yb, self.sub_h], stop=[xr, yt, self.sub_h], priority=10)
            
        # Ground strap via to bottom ground
        copper.AddBox(start=[xt_l + self.L3, 0.0, 0.0],
                      stop=[x_fl - self.L8, self.H2, self.sub_h], priority=10)
                      
        # Lumped 50-Ohm Port at feed line (y = 0)
        port_start = [x_fl, 0.0, 0.0]
        port_stop = [x_fr, 0.0, self.sub_h]
        port = FDTD.AddLumpedPort(1, 50.0, port_start, port_stop, 'z', 1.0, priority=20, edges2grid='xy')
        
        # Mesh (guaranteeing at least 60 cells/lambda in all dimensions)
        res = self.lambda0_mm / self.cells_per_wavelength
        air_pad = self.lambda0_mm * 0.45
        
        # Z lines
        nz_sub = 4
        z_sub = np.linspace(0.0, self.sub_h, nz_sub + 1)
        z_raw = np.concatenate([[-air_pad], z_sub, [self.sub_h + air_pad]])
        z_lines = self._smooth_mesh(z_raw, res, ratio=1.30)
        
        # X lines with air buffer
        x_pts = [-self.sub_w/2 - air_pad, -self.sub_w/2, self.sub_w/2, self.sub_w/2 + air_pad, xt_l, xt_r, x_fl, x_fr]
        for _, xl, xr, _, _ in boxes:
            x_pts.extend([xl, xr])
        x_lines = self._smooth_mesh(np.array(x_pts), res, ratio=1.30)
        
        # Y lines with air buffer
        y_pts = [self.sub_y_min - air_pad, self.sub_y_min, 0.0, self.H1, self.H1 + self.W1, self.sub_y_max, self.sub_y_max + air_pad]
        for _, _, _, yb, yt in boxes:
            y_pts.extend([yb, yt])
        y_lines = self._smooth_mesh(np.array(y_pts), res, ratio=1.30)
        
        mesh.AddLine('x', x_lines)
        mesh.AddLine('y', y_lines)
        mesh.AddLine('z', z_lines)
        
        nf2ff = FDTD.CreateNF2FFBox()
        return FDTD, CSX, [port], nf2ff

    def draw_layout(self, output_path: str):
        fig, (ax_plan, ax_side) = plt.subplots(1, 2, figsize=(14, 7),
                                               gridspec_kw={'width_ratios': [1.3, 1.0]},
                                               tight_layout=True)
        # Substrate
        ax_plan.add_patch(mpatches.Rectangle((-self.sub_w/2, self.sub_y_min), self.sub_w, self.sub_l,
                                            facecolor='#f8fafc', edgecolor='#94a3b8', linewidth=1.2, label='FR-4 Substrate'))
        # Ground plane
        boxes, (x_fl, x_fr), (xt_l, xt_r) = self._get_trace_boxes()
        ax_plan.add_patch(mpatches.Rectangle((-self.sub_w/2, self.sub_y_min), -self.sub_w/2 + (x_fl - self.L8), self.ground_length,
                                            facecolor='#cbd5e1', edgecolor='#64748b', linewidth=1.0, label='Ground Plane (Left)'))
        ax_plan.add_patch(mpatches.Rectangle((x_fr + self.L8, self.sub_y_min), self.sub_w/2 - (x_fr + self.L8), self.ground_length,
                                            facecolor='#cbd5e1', edgecolor='#64748b', linewidth=1.0, label='Ground Plane (Right)'))
                                            
        # Antenna traces
        for name, xl, xr, yb, yt in boxes:
            color = '#dc2626' if 'feed' in name else ('#b45309' if 'gnd' in name else '#0284c7')
            ax_plan.add_patch(mpatches.Rectangle((xl, yb), xr - xl, yt - yb, facecolor=color, edgecolor='#0f172a', linewidth=0.8))
            
        ax_plan.annotate('50 $\\Omega$ Feed Point', xy=((x_fl+x_fr)/2, 0), xytext=((x_fl+x_fr)/2 + 3.0, -5.0),
                         arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5), color='#dc2626', fontweight='bold', fontsize=9.5)
        ax_plan.annotate('Ground Shorting Block', xy=(xt_l + self.L3 + 1.0, 0.4), xytext=(xt_l - 2.0, -5.0),
                         arrowprops=dict(arrowstyle='->', color='#b45309', lw=1.5), color='#b45309', fontweight='bold', fontsize=9.5)
                         
        ax_plan.set_xlim([-self.sub_w/2 - 1.0, self.sub_w/2 + 1.0])
        ax_plan.set_ylim([self.sub_y_min - 2.0, self.sub_y_max + 1.0])
        ax_plan.set_xlabel('X Dimension (mm)', fontsize=10)
        ax_plan.set_ylabel('Y Dimension (mm)', fontsize=10)
        ax_plan.set_title('(a) TI SWRU120D 2.4 GHz IFA — Plan Layout\n(Footprint: 25.7 x 7.5 mm, L1=25.58mm, L2=16.40mm, H1=5.70mm, W1=1.21mm)', fontsize=10.5, fontweight='bold')
        ax_plan.grid(True, linestyle=':', alpha=0.6)
        
        # Side view
        ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, -0.02), self.sub_l, 0.02, facecolor='#b45309', label='Bottom GND Plane'))
        ax_side.add_patch(mpatches.Rectangle((self.sub_y_min, 0), self.sub_l, self.sub_h, facecolor='#dbeafe', edgecolor='#3b82f6', label=f'FR-4 Substrate ({self.sub_h:.2f} mm)'))
        ax_side.add_patch(mpatches.Rectangle((0, self.sub_h), self.H1 + self.W1, 0.035, facecolor='#0284c7', label='Top Cu Radiator'))
        ax_side.set_xlim([self.sub_y_min - 2.0, self.sub_y_max + 2.0])
        ax_side.set_ylim([-0.1, self.sub_h * 3.0])
        ax_side.set_xlabel('Y Dimension (mm)', fontsize=10)
        ax_side.set_ylabel('Z Dimension (mm)', fontsize=10)
        ax_side.set_title('(b) PCB Stackup Cross-Section', fontsize=10.5, fontweight='bold')
        ax_side.legend(loc='upper right', fontsize=8.5)
        ax_side.grid(True, linestyle=':', alpha=0.6)
        
        fig.suptitle(f"{self.title}\n{self.doc_reference}", fontsize=12, fontweight='bold')
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_3d_view(self, output_path: str):
        fig = plt.figure(figsize=(11, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        
        x0, x1 = -self.sub_w/2, self.sub_w/2
        y0, y1 = self.sub_y_min, self.sub_y_max
        z_gnd, z_top = 0.0, self.sub_h
        
        # Ground plane at bottom
        boxes, (x_fl, x_fr), (xt_l, xt_r) = self._get_trace_boxes()
        gnd_left = [[[x0, y0, z_gnd], [x_fl - self.L8, y0, z_gnd], [x_fl - self.L8, 0, z_gnd], [x0, 0, z_gnd]]]
        gnd_right = [[[x_fr + self.L8, y0, z_gnd], [x1, y0, z_gnd], [x1, 0, z_gnd], [x_fr + self.L8, 0, z_gnd]]]
        ax.add_collection3d(Poly3DCollection(gnd_left, facecolors='#b45309', alpha=0.85))
        ax.add_collection3d(Poly3DCollection(gnd_right, facecolors='#b45309', alpha=0.85))
        
        # Substrate
        sub_faces = [
            [[x0, y0, z_gnd], [x1, y0, z_gnd], [x1, y1, z_gnd], [x0, y1, z_gnd]],
            [[x0, y0, z_top], [x1, y0, z_top], [x1, y1, z_top], [x0, y1, z_top]]
        ]
        ax.add_collection3d(Poly3DCollection(sub_faces, facecolors='#dbeafe', edgecolors='#93c5fd', alpha=0.3))
        
        # Top traces
        for name, xl, xr, yb, yt in boxes:
            p = [[[xl, yb, z_top], [xr, yb, z_top], [xr, yt, z_top], [xl, yt, z_top]]]
            color = '#dc2626' if 'feed' in name else '#0284c7'
            ax.add_collection3d(Poly3DCollection(p, facecolors=color, alpha=1.0))
            
        ax.plot([(x_fl+x_fr)/2, (x_fl+x_fr)/2], [0, 0], [z_gnd, z_top], color='#dc2626', linewidth=3)
        ax.set_xlim([x0 - 2, x1 + 2])
        ax.set_ylim([y0 - 2, y1 + 2])
        ax.set_zlim([-0.5, z_top + 2.0])
        ax.view_init(elev=30, azim=-60)
        ax.set_title(f"{self.title}\n3D Isometric Model", fontsize=11, fontweight='bold')
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_interactive_3d(self, output_path: str):
        specs = self.get_proven_specs()
        boxes, (x_fl, x_fr), (xt_l, xt_r) = self._get_trace_boxes()
        boxes_js = []
        for name, xl, xr, yb, yt in boxes:
            boxes_js.append(f"{{ xl: {xl:.2f}, xr: {xr:.2f}, yb: {yb:.2f}, yt: {yt:.2f}, isFeed: {'true' if 'feed' in name else 'false'} }}")
        boxes_str = ",\n            ".join(boxes_js)
        
        w_left_gnd = self.sub_w / 2.0 + (x_fl - self.L8)
        w_right_gnd = self.sub_w / 2.0 - (x_fr + self.L8)
        pos_x_gnd_l = -self.sub_w / 2.0 + w_left_gnd / 2.0
        pos_x_gnd_r = self.sub_w / 2.0 - w_right_gnd / 2.0
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.title} — 3D Interactive Model</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc; }}
        #hud {{ position: absolute; top: 16px; left: 16px; background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px); padding: 16px 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.12); max-width: 340px; font-size: 12px; }}
        h2 {{ margin: 0 0 8px 0; font-size: 15px; color: #38bdf8; font-weight: 700; }}
        .metric {{ font-weight: 600; color: #f1f5f9; }}
        .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; background: #0284c7; color: white; margin-bottom: 8px; }}
        .proven {{ border-top: 1px solid rgba(255,255,255,0.1); margin-top: 10px; padding-top: 8px; font-size: 11px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <span class="badge">PROVEN REFERENCE MODEL</span>
        <h2>{self.title}</h2>
        <p>Reference: <span class="metric">{self.doc_reference}</span></p>
        <p>Operating Band: <span class="metric">{specs['operating_band']}</span></p>
        <p>Footprint: <span class="metric">{specs['antenna_dimensions']}</span></p>
        <p>Substrate: <span class="metric">{specs['substrate_material']}</span></p>
        <div class="proven">
            <strong style="color: #38bdf8;">TI Published Proven Metrics:</strong><br>
            • Reflection S11: {specs['proven_s11_reflection_db']}<br>
            • -10 dB Bandwidth: {specs['proven_10db_bandwidth_mhz']}<br>
            • Gain XZ / YZ: {specs['proven_gain_xz_plane_dbi']} / {specs['proven_gain_yz_plane_dbi']}<br>
            • Pattern: {specs['radiation_pattern']}
        </div>
    </div>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
        camera.position.set(30, -35, 30);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0, 0);
        controls.update();

        const light1 = new THREE.DirectionalLight(0xffffff, 1.0); light1.position.set(20, 20, 30); scene.add(light1);
        const light2 = new THREE.DirectionalLight(0x93c5fd, 0.6); light2.position.set(-20, -20, -10); scene.add(light2);
        scene.add(new THREE.AmbientLight(0x475569));

        // Substrate
        const subGeo = new THREE.BoxGeometry({self.sub_w:.2f}, {self.sub_l:.2f}, {self.sub_h:.2f});
        const subMat = new THREE.MeshStandardMaterial({{ color: 0x3b82f6, transparent: true, opacity: 0.35 }});
        const sub = new THREE.Mesh(subGeo, subMat);
        sub.position.set(0, {self.sub_y_min + self.sub_l/2:.2f}, {self.sub_h/2:.2f});
        scene.add(sub);

        // Ground Plane (bottom)
        const gndMat = new THREE.MeshStandardMaterial({{ color: 0xb45309, metalness: 0.8, roughness: 0.3 }});
        const gndL = new THREE.Mesh(new THREE.BoxGeometry({w_left_gnd:.2f}, {self.ground_length:.2f}, 0.05), gndMat);
        gndL.position.set({pos_x_gnd_l:.2f}, {-self.ground_length/2:.2f}, -0.02);
        scene.add(gndL);

        const gndR = new THREE.Mesh(new THREE.BoxGeometry({w_right_gnd:.2f}, {self.ground_length:.2f}, 0.05), gndMat);
        gndR.position.set({pos_x_gnd_r:.2f}, {-self.ground_length/2:.2f}, -0.02);
        scene.add(gndR);

        // Antenna traces
        const traceMat = new THREE.MeshStandardMaterial({{ color: 0x0284c7, metalness: 0.7, roughness: 0.3 }});
        const feedMat = new THREE.MeshStandardMaterial({{ color: 0xdc2626, metalness: 0.8, roughness: 0.2 }});
        const traces = [
            {boxes_str}
        ];
        traces.forEach(t => {{
            const w = t.xr - t.xl;
            const l = t.yt - t.yb;
            const geo = new THREE.BoxGeometry(w, l, 0.06);
            const m = new THREE.Mesh(geo, t.isFeed ? feedMat : traceMat);
            m.position.set(t.xl + w/2, t.yb + l/2, {self.sub_h:.2f} + 0.03);
            scene.add(m);
        }});

        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
        animate();
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)


# ==============================================================================
# Model Catalog & Factory
# ==============================================================================

SUPPORTED_ANTENNAS = {
    'ti_77ghz_rx': TI77GHzRXAntenna,
    'ti_24ghz_an043': TI24GHzAN043Antenna,
    'ti_24ghz_swru120d': TI24GHzSWRU120DAntenna
}

def get_antenna_model(antenna_name: str = 'ti_77ghz_rx', **kwargs) -> BaseAntennaModel:
    """Factory function to instantiate a proven antenna model."""
    name_clean = str(antenna_name).strip().lower()
    if name_clean in SUPPORTED_ANTENNAS:
        return SUPPORTED_ANTENNAS[name_clean](**kwargs)
    
    # Aliases
    if '77' in name_clean or 'swru553' in name_clean:
        return TI77GHzRXAntenna(**kwargs)
    elif 'an043' in name_clean or 'mifa' in name_clean:
        return TI24GHzAN043Antenna(**kwargs)
    elif 'swru120' in name_clean or 'dn007' in name_clean or 'ifa' in name_clean:
        return TI24GHzSWRU120DAntenna(**kwargs)
    else:
        raise ValueError(f"Unknown antenna '{antenna_name}'. Supported antennas: {list(SUPPORTED_ANTENNAS.keys())}")
