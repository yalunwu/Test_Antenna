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
from reference_geometry import GerberPCBGeometry, circle_points, rectangle

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
    def create_simulation(self, end_criteria: float = 1e-6, nr_ts: int = 500000, active_channels=1):
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
        """Subdivide gaps to max_res and remove numerical duplicates.

        This legacy builder does not enforce ratio; graded meshing and local
        convergence remain separate from the source-geometry corrections.
        """
        lines = np.unique(np.sort(lines))
        lines = lines[np.r_[True, np.diff(lines) > 1e-9]]
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
        self.A1_gcpw_len = 27.568 * 0.0254      # wider line before ground termination
        self.A2_trans_len = 11.802 * 0.0254     # wider line after ground termination
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
        self.via_drill_diameter = 5.9 * 0.0254
        self.via_pad_diameter = 13.77 * 0.0254
        self.gcpw_ground_w = 0.60  # cropped ground width, not specified in Table 7
        self.gcpw_lead_length = 0.70  # cropped input lead, not dimensioned in Figure 16
        self.geometry_revision = 'ti77-gcpw-ground-vias-v2'
        
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
        self.y_gcpw_start = -self.gcpw_lead_length
        self.y_gcpw_end = 0.0
        
        # Figure 16: A1 is the wider line under the coplanar ground,
        # followed by A2 beyond the ground termination. Both have width D2.
        self.y_trans_start = 0.0
        self.y_ground_end = self.A1_gcpw_len
        self.y_trans_end = self.A1_gcpw_len + self.A2_trans_len
        
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
            "proven_s11_reflection_db": "Not specified in Figures 18-20; -10 dB is an engineering goal",
            "proven_single_element_gain_dbi": "9.79-10.87 dB system gain in RX cuts at 76 GHz (Figures 18-20)",
            "proven_elevation_hpbw_deg": "21.20-25.26 deg at 76 GHz (E-plane)",
            "proven_azimuth_hpbw_deg": "62.89-75.43 deg at 76 GHz (H-plane)",
            "proven_array_directivity_dbi": "Not specified in the cited element-performance figures",
            "substrate_material": "Rogers RO3003 (0.127 mm / 5 mil, eps_r = 3.00)",
            "board_stackup": "4-Layer mmWave PCB (RO3003 + FR-4 core)"
        }

    def _get_gcpw_ground_polygons(self, xc):
        inner1 = self.D1_gcpw_w/2 + self.F1_gcpw_gap
        inner2 = self.D2_trans_w/2 + self.F2_taper_gap
        outer = max(inner1, inner2) + self.gcpw_ground_w
        right = np.array([[inner1, self.y_gcpw_start], [outer, self.y_gcpw_start],
                          [outer, self.y_ground_end], [inner2, self.y_ground_end],
                          [inner2, self.y_trans_start], [inner1, self.y_trans_start]])
        left = right * [-1, 1]
        return [left + [xc, 0], right + [xc, 0]]

    def _get_ground_vias(self):
        radius = self.via_pad_diameter/2
        low, high = self.y_gcpw_start + radius, self.y_ground_end - radius
        count = max(2, int(np.ceil((high-low)/self.H_via_pitch)) + 1)
        offset = max(self.D1_gcpw_w/2 + self.F1_gcpw_gap,
                     self.D2_trans_w/2 + self.F2_taper_gap) + self.gcpw_ground_w/2
        return [(float(xc + sign*offset), float(y)) for xc in self.channel_offsets_x
                for sign in (-1, 1) for y in np.linspace(low, high, count)]

    def create_simulation(self, end_criteria: float = 1e-6, nr_ts: int = 500000, active_channels=1):
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

        # 5. Top metallization and grounded coplanar transition.
        for xc in self.channel_offsets_x:
            copper.AddBox(start=[xc-self.D1_gcpw_w/2, self.y_gcpw_start, self.sub_h],
                          stop=[xc+self.D1_gcpw_w/2, self.y_gcpw_end, self.sub_h], priority=10)
            for polygon in self._get_gcpw_ground_polygons(xc):
                copper.AddLinPoly(polygon.T, 'z', self.sub_h, 0.0, priority=10)
        for x, y in self._get_ground_vias():
            # Solid PEC via barrel approximation; published drill and pad sizes.
            copper.AddCylinder([x, y, 0], [x, y, self.sub_h], self.via_drill_diameter/2, priority=11)
            copper.AddLinPoly(circle_points(x, y, self.via_pad_diameter/2).T,
                              'z', self.sub_h, 0.0, priority=11)
        for xc in self.channel_offsets_x:
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
            port_start = [xc - self.D1_gcpw_w/2, self.y_gcpw_start, 0.0]
            port_stop = [xc + self.D1_gcpw_w/2, self.y_gcpw_start, self.sub_h]
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
        for xc in self.channel_offsets_x:
            for polygon in self._get_gcpw_ground_polygons(xc):
                x_pts.extend(polygon[:, 0])
            x_pts.extend([xc-self.D2_trans_w/2, xc+self.D2_trans_w/2])
        for x, _ in self._get_ground_vias():
            x_pts.extend([x-self.via_drill_diameter/2, x, x+self.via_drill_diameter/2])
        x_lines = self._smooth_mesh(np.array(x_pts), res, ratio=1.35)
        
        # Y lines
        y_pts = [
            self.sub_y_min - self.lambda0_mm * 0.5, self.sub_y_min,
            0.0, self.y_gcpw_end, self.y_trans_end, self.y_stub_end,
            self.y_p1_start, self.y_p1_end, self.y_p2_start, self.y_p2_end,
            self.y_p3_start, self.y_p3_end, self.y_p4_start, self.y_p4_end,
            self.sub_y_max, self.sub_y_max + self.lambda0_mm * 0.5
        ]
        y_pts.extend([self.y_gcpw_start, self.y_ground_end])
        for _, y in self._get_ground_vias():
            y_pts.extend([y-self.via_drill_diameter/2, y, y+self.via_drill_diameter/2])
        y_lines = self._smooth_mesh(np.array(y_pts), res, ratio=1.35)
        
        mesh.SetLines('x', x_lines)
        mesh.SetLines('y', y_lines)
        mesh.SetLines('z', z_lines)
        
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
            ax_plan.add_patch(mpatches.Rectangle((xc - self.D1_gcpw_w/2, self.y_gcpw_start), self.D1_gcpw_w, self.gcpw_lead_length, facecolor='#dc2626'))
            ax_plan.add_patch(mpatches.Rectangle((xc - self.D2_trans_w/2, self.y_trans_start), self.D2_trans_w, self.y_trans_end-self.y_trans_start, facecolor='#ea580c'))
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
            
        for xc in self.channel_offsets_x:
            for polygon in self._get_gcpw_ground_polygons(xc):
                ax_plan.add_patch(mpatches.Polygon(polygon, facecolor='#94a3b8', edgecolor='#64748b'))
        for x, y in self._get_ground_vias():
            ax_plan.add_patch(mpatches.Circle((x, y), self.via_pad_diameter/2, facecolor='#64748b'))
            ax_plan.add_patch(mpatches.Circle((x, y), self.via_drill_diameter/2, facecolor='#e2e8f0'))
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
        
        fig.suptitle(f"{self.title}\nReference radiator dimensions on a cropped board", fontsize=12, fontweight='bold')
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def draw_3d_view(self, output_path: str):
        fig = plt.figure(figsize=(11, 8), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        
        x0, x1 = -self.sub_w/2, self.sub_w/2
        y0, y1 = self.sub_y_min, self.sub_y_max
        z_gnd, z_rf, z_core = 0.0, self.sub_h, -self.h_core
        
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
            z_cu = z_rf
            p = [[[xc - xw/2, ys, z_cu], [xc + xw/2, ys, z_cu], [xc + xw/2, ye, z_cu], [xc - xw/2, ye, z_cu]]]
            ax.add_collection3d(Poly3DCollection(p, facecolors=color, edgecolors='#78350f', alpha=1.0, linewidth=0.8))
            
        for ch_idx, xc in enumerate(self.channel_offsets_x):
            add_strip(xc, self.D1_gcpw_w, self.y_gcpw_start, self.y_gcpw_end, '#dc2626')
            add_strip(xc, self.D2_trans_w, self.y_trans_start, self.y_trans_end, '#ea580c')
            add_strip(xc, self.F3_feed_w, self.y_stub_start, self.y_stub_end, '#f59e0b')
            add_strip(xc, self.E1_patch14_w, self.y_p1_start, self.y_p1_end, '#2563eb')
            add_strip(xc, self.F3_feed_w, self.y_i1_start, self.y_i1_end, '#10b981')
            add_strip(xc, self.E2_patch23_w, self.y_p2_start, self.y_p2_end, '#4f46e5')
            add_strip(xc, self.F3_feed_w, self.y_i2_start, self.y_i2_end, '#10b981')
            add_strip(xc, self.E2_patch23_w, self.y_p3_start, self.y_p3_end, '#4f46e5')
            add_strip(xc, self.F3_feed_w, self.y_i3_start, self.y_i3_end, '#10b981')
            add_strip(xc, self.E1_patch14_w, self.y_p4_start, self.y_p4_end, '#2563eb')
            
            ax.plot([xc, xc], [self.y_gcpw_start]*2, [z_gnd, z_rf], color='#dc2626', linewidth=3)
            ax.text(xc, -0.6, z_rf/2, f'P{ch_idx+1}', color='#dc2626', fontweight='bold', fontsize=8)

        ax.set_xlim([-self.sub_w/2 - 0.5, self.sub_w/2 + 0.5])
        ax.set_ylim([self.sub_y_min - 0.5, self.sub_y_max + 0.5])
        for xc in self.channel_offsets_x:
            for polygon in self._get_gcpw_ground_polygons(xc):
                ax.add_collection3d(Poly3DCollection([np.c_[polygon, np.full(len(polygon), z_rf)]], facecolor='#64748b'))
        for x, y in self._get_ground_vias():
            ax.plot([x, x], [y, y], [0, z_rf], color='#64748b', linewidth=2)
            pad = circle_points(x, y, self.via_pad_diameter/2)
            ax.add_collection3d(Poly3DCollection([np.c_[pad, np.full(len(pad), z_rf)]], facecolor='#64748b'))
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
        const subGeo = new THREE.BoxGeometry({self.sub_w:.2f}, {self.sub_l:.2f}, {self.sub_h:.4f});
        const subMat = new THREE.MeshStandardMaterial({{ color: 0x3b82f6, transparent: true, opacity: 0.35 }});
        const sub = new THREE.Mesh(subGeo, subMat);
        sub.position.set(0, {self.sub_y_min + self.sub_l/2:.2f}, {self.sub_h/2:.4f});
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
            m.position.set(x, y + l/2, {self.sub_h:.4f});
            scene.add(m);
        }}

        const offsets = {self.channel_offsets_x};
        offsets.forEach(xc => {{
            addBox(xc, {self.y_gcpw_start:.4f}, {self.D1_gcpw_w:.4f}, {self.gcpw_lead_length:.4f}, cuMat);
            addBox(xc, {self.y_trans_start:.3f}, {self.D2_trans_w:.4f}, {self.y_trans_end-self.y_trans_start:.4f}, cuMat);
            addBox(xc, {self.y_stub_start:.3f}, {self.F3_feed_w:.3f}, {self.B1_stub_len:.3f}, cuMat);
            addBox(xc, {self.y_p1_start:.3f}, {self.E1_patch14_w:.3f}, {self.C1_patch14_len:.3f}, patchMat);
            addBox(xc, {self.y_i1_start:.3f}, {self.F3_feed_w:.3f}, {self.B2_inter1_len:.3f}, cuMat);
            addBox(xc, {self.y_p2_start:.3f}, {self.E2_patch23_w:.3f}, {self.C2_patch23_len:.3f}, patchMat);
            addBox(xc, {self.y_i2_start:.3f}, {self.F3_feed_w:.3f}, {self.B3_inter2_len:.3f}, cuMat);
            addBox(xc, {self.y_p3_start:.3f}, {self.E2_patch23_w:.3f}, {self.C2_patch23_len:.3f}, patchMat);
            addBox(xc, {self.y_i3_start:.3f}, {self.F3_feed_w:.3f}, {self.B4_inter3_len:.3f}, cuMat);
            addBox(xc, {self.y_p4_start:.3f}, {self.E1_patch14_w:.3f}, {self.C1_patch14_len:.3f}, patchMat);
        }});

        const groundShapes = {[[p.tolist() for p in self._get_gcpw_ground_polygons(x)] for x in self.channel_offsets_x]};
        groundShapes.flat().forEach(points => {{
            const shape = new THREE.Shape(); points.forEach((p, i) => i ? shape.lineTo(...p) : shape.moveTo(...p)); shape.closePath();
            const m = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({{color:0x64748b, side:THREE.DoubleSide}}));
            m.position.z={self.sub_h:.4f}; scene.add(m);
        }});
        const groundVias = {[list(p) for p in self._get_ground_vias()]};
        groundVias.forEach(p => {{
            const geo = new THREE.CylinderGeometry({self.via_drill_diameter/2:.6f}, {self.via_drill_diameter/2:.6f}, {self.sub_h:.4f}, 16);
            geo.rotateX(Math.PI/2); const m = new THREE.Mesh(geo, gndMat); m.position.set(p[0],p[1],{self.sub_h/2:.4f}); scene.add(m);
            const pad = new THREE.Mesh(new THREE.CircleGeometry({self.via_pad_diameter/2:.6f},32),cuMat);
            pad.position.set(p[0],p[1],{self.sub_h:.4f}); scene.add(pad);
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

class TI24GHzAN043Antenna(GerberPCBGeometry, BaseAntennaModel):
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
        # CC2511 USB reference README: L1-L2 spacing 0.25 mm.
        self.rf_ground_depth = min(0.25, self.sub_h)
        self.radiator_edge_margin = self.D2
        self.coplanar_ground = False
        self._initialize_outline('cc2511', align_left=-self.antenna_footprint_w/2 + self.D1)
        # Ground strap is 0.9 mm wide; 10-mil drill from the reference drill table.
        self.ground_vias = [(float(self.radiator_polygons[0][:, 0].min()) + self.W1/2,
                             -self.D4/2, 0.254/2)]
        self.geometry_revision = 'ti-gerber-l2-ground-v2'

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







# ==============================================================================
# 3. TI 2.4 GHz Inverted-F Antenna from SWRU120D (DN0007)
# ==============================================================================

class TI24GHzSWRU120DAntenna(GerberPCBGeometry, BaseAntennaModel):
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
        # CC2430DB manufacturing README: L1-L2 dielectric spacing 0.321 mm.
        # Keep the application note's 1 mm overall FR4 baseline.
        self.rf_ground_depth = min(0.321, self.sub_h)
        self.radiator_edge_margin = self.H9
        self.coplanar_ground = True
        self._initialize_outline('cc2430')
        # Stitch the coplanar grounds to L2 behind the two source ground feet.
        # 0.30 mm stitch diameter is a board-environment modeling parameter.
        shift = float(self.feed_center[0])
        source_origin_x = self.reference_outline['source_origin'][0]
        self.ground_vias = [((x-source_origin_x)*0.00254 + shift, -0.4, 0.15)
                            for x in (17087, 17904)]
        self.geometry_revision = 'ti-gerber-l2-ground-v2'

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







# ==============================================================================
# 4. MDPI 24 GHz 4x1 Microstrip Patch Array (Electronics 2023, 12(7), 1512)
# ==============================================================================

class MDPI24GHzPatchArrayAntenna(BaseAntennaModel):
    """
    24 GHz 4x1 Microstrip Square Patch Array Antenna with Shunt-Connected Series Tapered Feed Network,
    with optional dual tilted metallic plates for high gain enhancement.
    Direct implementation of MDPI Electronics 2023, 12(7), 1512
    (https://doi.org/10.3390/electronics12071512)
    "Gain Enhancement of Microstrip Patch Array Antennas Using Two Metallic Plates for 24 GHz Radar Applications"
    by Junho Yeo and Jong-Ig Lee.

    Stackup & Substrate:
      - Layer 1 (Top Cu): 4x1 Inset-Fed Square Patch Array with 1:5:5:1 Tapered Feed Network
      - Dielectric: 0.254 mm (10 mil) HF-350F (eps_r = 3.50, tan_d = 0.0029)
      - Layer 2: Continuous Copper Ground Plane (30.0 mm x 12.0 mm)
      - (Optional) Two 50 mm x 30 mm Copper Plates flared at 70° for +7 dB gain boost
    """
    def __init__(self,
                 with_plates: bool = False,
                 plate_length: float = 50.0,
                 plate_angle_deg: float = 70.0,
                 cells_per_wavelength: float = 60.0):
        self.with_plates = with_plates
        self.plate_length = float(plate_length)
        self.plate_angle_deg = float(plate_angle_deg)
        
        name_str = "mdpi_24ghz_plates" if with_plates else "mdpi_24ghz_array"
        title_str = ("MDPI 24 GHz 4x1 Array with 70° Metallic Plates (Electronics 12/7/1512)"
                     if with_plates else
                     "MDPI 24 GHz 4x1 Microstrip Patch Array (Electronics 12/7/1512)")
        
        super().__init__(
            name=name_str,
            title=title_str,
            doc_reference="MDPI Electronics 2023, 12(7), 1512 (https://doi.org/10.3390/electronics12071512)",
            f_center=24.125e9,
            f_start=24.00e9,
            f_stop=24.25e9,
            f_wide_start=23.50e9,
            f_wide_stop=24.75e9,
            cells_per_wavelength=max(60.0, float(cells_per_wavelength))
        )
        self.sub_w = 30.0   # mm (Wg2)
        self.sub_l = 12.0   # mm (Lg2)
        self.sub_h = 0.254  # mm (10 mil)
        self.sub_eps_r = 3.50
        self.sub_tan_d = 0.0029
        self.sub_kappa = 2 * np.pi * self.f_center * EPS0 * self.sub_eps_r * self.sub_tan_d
        
        # Dimensions from paper Section 2 & Figures 1, 3, 5 (mm)
        self.L1 = 3.22      # Square patch length & width
        self.l2 = 7.49      # Element spacing (~1 guided wavelength)
        self.l1 = 1.87      # QWT length (1/4 * l2)
        self.l3 = 3.75      # Outer bus line segment
        self.w1 = 0.54      # 50-Ohm feed line width
        self.w2 = 0.125     # 100-Ohm branch lines width
        self.w3 = 0.16      # QWT 1 width (91.3 Ohm)
        self.w4 = 0.65      # QWT 2 width (44.7 Ohm)
        self.r1 = 0.45      # Port 1 matching circular disk radius
        self.wis2 = 0.25    # Inset notch gap width
        self.lis2 = 1.10    # Inset depth into patch
        
        # Vertical placement
        self.y_bus = -3.00
        self.y_bot = self.y_bus + self.l1   # -1.13 mm
        self.y_top = self.y_bot + self.L1   # +2.09 mm
        self.y_port = self.y_bus + self.l1  # -1.13 mm
        
        # 4 Patch centers along x
        self.x_patches = [-1.5 * self.l2, -0.5 * self.l2, 0.5 * self.l2, 1.5 * self.l2]
        
        # Metallic plates geometry
        self.theta_rad = np.deg2rad(self.plate_angle_deg)
        self.dy_plate = self.plate_length * np.cos(self.theta_rad) if self.with_plates else 0.0
        self.dz_plate = self.plate_length * np.sin(self.theta_rad) if self.with_plates else 0.0
        if not 0 < self.plate_angle_deg < 90 or self.plate_length < 0:
            raise ValueError('Plate angle must be between 0 and 90 degrees and length nonnegative')
        self.plate_copper_thickness = 0.036  # mm; source tape, represented by a thin PEC surface
        self.geometry_revision = 'mdpi-circular-pad-thin-plates-v2'

    def get_proven_specs(self) -> dict:
        if self.with_plates:
            return {
                "source_doc": "MDPI Electronics 2023, 12(7), 1512 (Figures 7–10 & Table 1)",
                "operating_band": "24.00 – 24.25 GHz K-Band Radar",
                "nominal_center_freq_ghz": 24.125,
                "proven_s11_reflection_db": "< -20.0 dB at 24.125 GHz (VSWR < 2: 23.767 – 24.414 GHz)",
                "proven_10db_bandwidth_mhz": "647 MHz (23.767 – 24.414 GHz, 2.69%)",
                "proven_peak_gain_dbi": "18.29 dBi (simulated) / 16.8 dBi (measured) [+7.2 dB gain boost]",
                "proven_elevation_hpbw_deg": "Narrow focused elevation beam (flared horn effect)",
                "proven_azimuth_hpbw_deg": "26.6° (H-Plane / z-x plane)",
                "proven_sidelobe_level_db": "-26.65 dB (simulated) / -33.66 dB (measured)",
                "metallic_plates": f"Two {self.plate_length:.1f} mm copper plates at {self.plate_angle_deg:.1f}° tilt (aperture {self.sub_l+2*self.dy_plate:.2f} mm x {self.sub_w:.2f} mm)",
                "antenna_dimensions": f"{self.sub_w:.1f} mm x {self.sub_l:.1f} mm PCB with {self.plate_length:.1f} mm flared plates",
                "substrate_material": "HF-350F (eps_r = 3.50, tan_d = 0.0029, h = 0.254 mm / 10 mil)"
            }
        else:
            return {
                "source_doc": "MDPI Electronics 2023, 12(7), 1512 (Figures 5 & 6)",
                "operating_band": "24.00 – 24.25 GHz K-Band Radar",
                "nominal_center_freq_ghz": 24.125,
                "proven_s11_reflection_db": "< -20.0 dB at 24.125 GHz (VSWR < 2: 23.836 – 24.402 GHz)",
                "proven_10db_bandwidth_mhz": "566 MHz (23.836 – 24.402 GHz, 2.35%)",
                "proven_peak_gain_dbi": "11.11 dBi (simulated) / 9.8 dBi (measured)",
                "proven_elevation_hpbw_deg": "75.5° (E-Plane / y-z plane)",
                "proven_azimuth_hpbw_deg": "26.6° (H-Plane / z-x plane)",
                "proven_sidelobe_level_db": "-21.3 dB (simulated) / -17.83 dB (measured)",
                "antenna_dimensions": f"{self.sub_w:.1f} mm x {self.sub_l:.1f} mm compact PCB",
                "substrate_material": "HF-350F (eps_r = 3.50, tan_d = 0.0029, h = 0.254 mm / 10 mil)"
            }

    def _get_trace_boxes(self):
        """Return all copper rectangular trace blocks."""
        boxes = []
        
        # 1. Four Inset-fed Patches
        for idx, xc in enumerate(self.x_patches):
            # Left wing of patch
            boxes.append((f'patch_{idx+1}_left', xc - self.L1/2, xc - self.w2/2 - self.wis2, self.y_bot, self.y_top))
            # Right wing of patch
            boxes.append((f'patch_{idx+1}_right', xc + self.w2/2 + self.wis2, xc + self.L1/2, self.y_bot, self.y_top))
            # Upper head of patch above inset
            boxes.append((f'patch_{idx+1}_head', xc - self.w2/2 - self.wis2, xc + self.w2/2 + self.wis2, self.y_bot + self.lis2, self.y_top))
            # Inset tongue
            boxes.append((f'patch_{idx+1}_tongue', xc - self.w2/2, xc + self.w2/2, self.y_bot, self.y_bot + self.lis2))
            # Vertical branch from bus
            boxes.append((f'branch_{idx+1}', xc - self.w2/2, xc + self.w2/2, self.y_bus, self.y_bot))
            
        # 2. Port 1 feed line. The circular pad is a polygon, never a box.
        boxes.append(('feed_port1', -self.w1/2, self.w1/2, self.y_bus, self.y_port))
        
        # 3. Horizontal Bus (Right half)
        boxes.append(('bus_r1', 0.0, self.l1, self.y_bus - self.w2/2, self.y_bus + self.w2/2))
        boxes.append(('bus_r2_qwt1', self.l1, 2*self.l1, self.y_bus - self.w3/2, self.y_bus + self.w3/2))
        boxes.append(('bus_r3', 2*self.l1, 3*self.l1, self.y_bus - self.w2/2, self.y_bus + self.w2/2))
        boxes.append(('bus_r4_qwt2', 3*self.l1, 4*self.l1, self.y_bus - self.w4/2, self.y_bus + self.w4/2))
        boxes.append(('bus_r5', 4*self.l1, self.x_patches[3], self.y_bus - self.w2/2, self.y_bus + self.w2/2))
        
        # 4. Horizontal Bus (Left half)
        boxes.append(('bus_l1', -self.l1, 0.0, self.y_bus - self.w2/2, self.y_bus + self.w2/2))
        boxes.append(('bus_l2_qwt1', -2*self.l1, -self.l1, self.y_bus - self.w3/2, self.y_bus + self.w3/2))
        boxes.append(('bus_l3', -3*self.l1, -2*self.l1, self.y_bus - self.w2/2, self.y_bus + self.w2/2))
        boxes.append(('bus_l4_qwt2', -4*self.l1, -3*self.l1, self.y_bus - self.w4/2, self.y_bus + self.w4/2))
        boxes.append(('bus_l5', self.x_patches[0], -4*self.l1, self.y_bus - self.w2/2, self.y_bus + self.w2/2))
        
        return boxes

    def _get_matching_disk(self):
        return circle_points(0.0, self.y_port, self.r1)

    def _get_plate_facets(self, y_lines, z_lines):
        """Connected, zero-thickness PEC staircase on the actual Yee grid.

        A 36 um slanted volume can disappear under cell sampling. Axis-aligned
        sheet faces preserve a continuous reflector without inventing 0.8 mm
        metal. Surface-position error shrinks with the y/z grid spacing.
        The acrylic laboratory fixture is not part of this ideal antenna model.
        """
        if not self.with_plates or self.plate_length == 0:
            return []
        y = np.asarray(y_lines)
        z = np.asarray(z_lines)
        z = z[(z >= -1e-9) & (z <= self.dz_plate+1e-9)]
        facets = []
        previous = self.sub_l/2
        for bottom, top in zip(z[:-1], z[1:]):
            ideal = self.sub_l/2 + (bottom+top)/2 / np.tan(self.theta_rad)
            current = float(y[np.argmin(abs(y-ideal))])
            current = min(max(current, previous), self.sub_l/2+self.dy_plate)
            if current > previous + 1e-10:
                facets.append((previous, current, bottom, bottom))
            facets.append((current, current, bottom, top))
            previous = current
        tip = self.sub_l/2 + self.dy_plate
        if tip > previous + 1e-10:
            facets.append((previous, tip, self.dz_plate, self.dz_plate))
        return facets

    def create_simulation(self, end_criteria: float = 1e-6, nr_ts: int = 500000, active_channels=1):
        """Construct CSXCAD model and openEMS FDTD configuration."""
        FDTD = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
        fc = 2.5e9  # 2.5 GHz Gaussian pulse bandwidth around 24.125 GHz
        FDTD.SetGaussExcite(self.f_center, fc)
        FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(self.unit)
        
        # 1. Materials
        sub = CSX.AddMaterial('hf350f', epsilon=self.sub_eps_r, kappa=self.sub_kappa)
        copper = CSX.AddMetal('copper')
        
        # 2. Substrate
        sub.AddBox([-self.sub_w/2, -self.sub_l/2, 0.0],
                   [self.sub_w/2, self.sub_l/2, self.sub_h], priority=1)
                   
        # 3. Ground plane at z = 0
        copper.AddBox([-self.sub_w/2, -self.sub_l/2, 0.0],
                      [self.sub_w/2, self.sub_l/2, 0.0], priority=10)
                      
        # 4. Top copper traces & patches
        boxes = self._get_trace_boxes()
        for name, xl, xr, yb, yt in boxes:
            copper.AddBox([xl, yb, self.sub_h], [xr, yt, self.sub_h], priority=10)
            
        # Matching disk at Port 1
        copper.AddLinPoly(self._get_matching_disk().T, 'z', self.sub_h, 0.0, priority=11)
            
        # 6. Lumped 50-Ohm Port at coaxial feed location
        port_start = [0.0, self.y_port, 0.0]
        port_stop = [0.0, self.y_port, self.sub_h]
        port = FDTD.AddLumpedPort(1, 50.0, port_start, port_stop, 'z', 1.0, priority=20, edges2grid='xy')
        
        # 7. Mesh Definition (guaranteeing >= 60 cells/lambda everywhere)
        res = self.lambda0_mm / self.cells_per_wavelength
        air_pad = self.lambda0_mm * 0.45
        
        # Z lines
        nz_sub = 4
        z_sub = np.linspace(0.0, self.sub_h, nz_sub + 1)
        z_top_limit = (self.dz_plate + air_pad) if self.with_plates else (self.sub_h + air_pad)
        z_features = [-air_pad] + list(z_sub) + ([self.dz_plate] if self.with_plates else []) + [z_top_limit]
        z_lines = self._smooth_mesh(np.array(z_features), res, ratio=1.30)
        
        # X lines
        x_features = [-self.sub_w/2 - air_pad, -self.sub_w/2, self.sub_w/2, self.sub_w/2 + air_pad]
        for _, xl, xr, _, _ in boxes:
            x_features.extend([xl, xr])
        x_features.extend([-self.r1, -self.r1/2, 0.0, self.r1/2, self.r1])
        x_lines = self._smooth_mesh(np.array(x_features), res, ratio=1.30)
        
        # Y lines
        y_bot_limit = (-self.sub_l/2 - self.dy_plate - air_pad) if self.with_plates else (-self.sub_l/2 - air_pad)
        y_top_limit = (self.sub_l/2 + self.dy_plate + air_pad) if self.with_plates else (self.sub_l/2 + air_pad)
        y_features = [y_bot_limit, -self.sub_l/2, self.sub_l/2, y_top_limit]
        if self.with_plates:
            y_features.extend([
                -self.sub_l/2 - self.dy_plate,
                self.sub_l/2 + self.dy_plate,
            ])
        for _, _, _, yb, yt in boxes:
            y_features.extend([yb, yt])
        y_features.extend(self.y_port + np.array([-self.r1, -self.r1/2, 0, self.r1/2, self.r1]))
        y_lines = self._smooth_mesh(np.array(y_features), res, ratio=1.30)
        
        mesh.AddLine('x', x_lines)
        mesh.AddLine('y', y_lines)
        mesh.AddLine('z', z_lines)

        for yl, yr, zl, zr in self._get_plate_facets(mesh.GetLines('y'), mesh.GetLines('z')):
            copper.AddBox([-self.sub_w/2, yl, zl], [self.sub_w/2, yr, zr], priority=15)
            # A zero-thickness plane must use the exact grid coordinate.
            # Negating the positive grid is not bit-identical to the negative
            # grid after smoothing and can leave entire PEC faces unused.
            native_y = mesh.GetLines('y')
            mirror_left = float(native_y[np.argmin(abs(native_y+yr))])
            mirror_right = float(native_y[np.argmin(abs(native_y+yl))])
            copper.AddBox([-self.sub_w/2, mirror_left, zl],
                          [self.sub_w/2, mirror_right, zr], priority=15)
        
        # 8. NF2FF
        nf2ff = FDTD.CreateNF2FFBox()
        return FDTD, CSX, [port], nf2ff

    def draw_layout(self, output_path: str):
        """Generate high-resolution 2D engineering layout diagram."""
        fig, (ax_plan, ax_side) = plt.subplots(1, 2, figsize=(15, 6),
                                               gridspec_kw={'width_ratios': [1.4, 0.9]},
                                               tight_layout=True)
        # Plan view
        sub_rect = mpatches.Rectangle((-self.sub_w/2, -self.sub_l/2), self.sub_w, self.sub_l,
                                     facecolor='#e2e8f0', edgecolor='#64748b', linewidth=1.5,
                                     label='HF-350F Substrate (eps_r=3.50)')
        ax_plan.add_patch(sub_rect)
        
        boxes = self._get_trace_boxes()
        for name, xl, xr, yb, yt in boxes:
            if 'disk' in name:
                continue
            r = mpatches.Rectangle((xl, yb), xr - xl, yt - yb,
                                   facecolor='#d97706', edgecolor='#b45309', linewidth=0.8)
            ax_plan.add_patch(r)
            
        # Disk
        disk = mpatches.Circle((0.0, self.y_port), self.r1, facecolor='#b45309', edgecolor='#78350f', linewidth=1.0)
        ax_plan.add_patch(disk)
        
        # Show plate projections if enabled
        if self.with_plates:
            plate_top = mpatches.Rectangle((-self.sub_w/2, self.sub_l/2), self.sub_w, self.dy_plate,
                                           facecolor='#fbbf24', edgecolor='#b45309', linestyle='--', alpha=0.35,
                                           label=f'Top Flared Plate ({self.plate_length:.0f} mm @ {self.plate_angle_deg:.0f}°)')
            plate_bot = mpatches.Rectangle((-self.sub_w/2, -self.sub_l/2 - self.dy_plate), self.sub_w, self.dy_plate,
                                           facecolor='#fbbf24', edgecolor='#b45309', linestyle='--', alpha=0.35,
                                           label=f'Bottom Flared Plate ({self.plate_length:.0f} mm @ {self.plate_angle_deg:.0f}°)')
            ax_plan.add_patch(plate_top)
            ax_plan.add_patch(plate_bot)
            
        # Annotations
        ax_plan.annotate(f'Wg2 = {self.sub_w:.1f} mm', xy=(0, self.sub_l/2 + (self.dy_plate if self.with_plates else 0) + 0.8),
                         ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e293b')
        ax_plan.annotate(f'Lg2 = {self.sub_l:.1f} mm', xy=(-self.sub_w/2 - 0.8, 0),
                         ha='right', va='center', fontsize=10, fontweight='bold', color='#1e293b', rotation=90)
        ax_plan.annotate(f'L1 = {self.L1:.2f} mm', xy=(self.x_patches[0], self.y_top + 0.6),
                         ha='center', va='bottom', fontsize=9, color='#0369a1')
        ax_plan.annotate(f'l2 = {self.l2:.2f} mm', xy=((self.x_patches[0] + self.x_patches[1])/2, self.y_top + 0.6),
                         ha='center', va='bottom', fontsize=8, color='#047857')
        ax_plan.annotate('Port 1 (50-Ohm)', xy=(0, self.y_port - 0.8),
                         ha='center', va='top', fontsize=8, fontweight='bold', color='#b91c1c')
                         
        y_lim_extra = (self.dy_plate + 2.0) if self.with_plates else 2.5
        ax_plan.set_xlim(-self.sub_w/2 - 3.0, self.sub_w/2 + 3.0)
        ax_plan.set_ylim(-self.sub_l/2 - y_lim_extra, self.sub_l/2 + y_lim_extra)
        ax_plan.set_aspect('equal')
        title_suffix = " with Dual 70° Flared Metallic Plates" if self.with_plates else ""
        ax_plan.set_title(f"MDPI 24 GHz 4x1 Microstrip Patch Array{title_suffix}\n(MDPI Electronics 2023, 12(7), 1512)",
                          fontsize=11, fontweight='bold')
        ax_plan.set_xlabel('X Dimension (mm)')
        ax_plan.set_ylabel('Y Dimension (mm)')
        ax_plan.legend(loc='lower left', fontsize=8)
        ax_plan.grid(True, linestyle=':', alpha=0.5)
        
        # Side view (Stackup & plate flare)
        ax_side.set_title("PCB Stackup & Flared Plate Cross-Section (YZ)", fontsize=11, fontweight='bold')
        ax_side.add_patch(mpatches.Rectangle((-self.sub_l/2, 0.0), self.sub_l, self.sub_h,
                                            facecolor='#e2e8f0', edgecolor='#64748b', linewidth=1.2,
                                            label=f'HF-350F ({self.sub_h*1000:.0f} um / 10 mil)'))
        ax_side.plot([-self.sub_l/2, self.sub_l/2], [0.0, 0.0], color='#b45309', linewidth=3.0, label='Bottom Ground (35 um Cu)')
        ax_side.plot([self.y_bot, self.y_top], [self.sub_h, self.sub_h], color='#d97706', linewidth=3.0, label='Top Cu Patches (35 um Cu)')
        
        if self.with_plates:
            # Top plate
            ax_side.plot([self.sub_l/2, self.sub_l/2 + self.dy_plate], [0.0, self.dz_plate],
                         color='#b45309', linewidth=3.5, label=f'Copper Plate (50 mm @ {self.plate_angle_deg:.0f}°)')
            # Bottom plate
            ax_side.plot([-self.sub_l/2, -self.sub_l/2 - self.dy_plate], [0.0, self.dz_plate],
                         color='#b45309', linewidth=3.5)
            ax_side.annotate(f'Aperture = {2*(self.sub_l/2 + self.dy_plate):.1f} mm',
                             xy=(0, self.dz_plate + 2.0), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#b45309')
            ax_side.set_xlim(-self.sub_l/2 - self.dy_plate - 5.0, self.sub_l/2 + self.dy_plate + 5.0)
            ax_side.set_ylim(-3.0, self.dz_plate + 6.0)
        else:
            ax_side.set_xlim(-self.sub_l/2 - 2.0, self.sub_l/2 + 2.0)
            ax_side.set_ylim(-0.8, 1.8)
            
        ax_side.set_xlabel('Y Dimension (mm)')
        ax_side.set_ylabel('Z Dimension (mm)')
        ax_side.legend(loc='upper right', fontsize=8)
        ax_side.grid(True, linestyle=':', alpha=0.5)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def draw_3d_view(self, output_path: str):
        """Generate publication-ready 3D isometric perspective diagram."""
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Substrate
        z_top = self.sub_h
        xx, yy = np.meshgrid([-self.sub_w/2, self.sub_w/2], [-self.sub_l/2, self.sub_l/2])
        ax.plot_surface(xx, yy, np.full_like(xx, z_top), color='#94a3b8', alpha=0.30)
        ax.plot_surface(xx, yy, np.zeros_like(xx), color='#b45309', alpha=0.75)
        
        # Patches and traces
        boxes = self._get_trace_boxes()
        for name, xl, xr, yb, yt in boxes:
            px, py = np.meshgrid([xl, xr], [yb, yt])
            ax.plot_surface(px, py, np.full_like(px, z_top + 0.02), color='#d97706', alpha=0.90)
            
        disk = self._get_matching_disk()
        ax.add_collection3d(Poly3DCollection([np.c_[disk, np.full(len(disk), z_top)]], facecolor='#d97706'))
        # Plates
        if self.with_plates:
            # Top plate
            x_pl = np.array([-self.sub_w/2, self.sub_w/2])
            y_top_pl = np.array([self.sub_l/2, self.sub_l/2 + self.dy_plate])
            X_top, Y_top = np.meshgrid(x_pl, y_top_pl)
            Z_top = (Y_top - self.sub_l/2) / self.dy_plate * self.dz_plate
            ax.plot_surface(X_top, Y_top, Z_top, color='#f59e0b', alpha=0.60)
            
            # Bottom plate
            y_bot_pl = np.array([-self.sub_l/2, -self.sub_l/2 - self.dy_plate])
            X_bot, Y_bot = np.meshgrid(x_pl, y_bot_pl)
            Z_bot = (-Y_bot - self.sub_l/2) / self.dy_plate * self.dz_plate
            ax.plot_surface(X_bot, Y_bot, Z_bot, color='#f59e0b', alpha=0.60)
            
        title_suffix = " with 70° Flared Metallic Plates" if self.with_plates else ""
        ax.set_title(f"MDPI 24 GHz 4x1 Microstrip Patch Array{title_suffix}\n(MDPI Electronics 2023, 12(7), 1512)",
                     fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.view_init(elev=28, azim=-55)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def draw_interactive_3d(self, output_path: str):
        """Generate interactive 3D WebGL Three.js HTML model with dark-theme HUD."""
        boxes = self._get_trace_boxes()
        boxes_js = []
        for name, xl, xr, yb, yt in boxes:
            xc = (xl + xr) / 2.0
            yc = (yb + yt) / 2.0
            xw = xr - xl
            yw = yt - yb
            boxes_js.append(f"{{ name: '{name}', x: {xc:.4f}, y: {yc:.4f}, w: {xw:.4f}, l: {yw:.4f} }}")
        boxes_str = ",\n        ".join(boxes_js)
        
        specs = self.get_proven_specs()
        gain_str = specs.get('proven_peak_gain_dbi', '18.29 dBi')
        plates_note = f"Dual Flared Copper Plates: Length {self.plate_length:.0f} mm, Tilt {self.plate_angle_deg:.0f}°" if self.with_plates else "Bare Microstrip Patch Array (Compact)"
        
        plates_code = ""
        if self.with_plates:
            plates_code = f"""
        // Dual Flared Metallic Plates
        const plateMat = new THREE.MeshStandardMaterial({{ color: 0xf59e0b, metalness: 0.9, roughness: 0.25, side: THREE.DoubleSide }});
        
        // Top Plate
        const topGeo = new THREE.BufferGeometry();
        const topVerts = new Float32Array([
            -subW/2,  subL/2, 0.0,
             subW/2,  subL/2, 0.0,
             subW/2,  subL/2 + {self.dy_plate:.4f}, {self.dz_plate:.4f},
            -subW/2,  subL/2, 0.0,
             subW/2,  subL/2 + {self.dy_plate:.4f}, {self.dz_plate:.4f},
            -subW/2,  subL/2 + {self.dy_plate:.4f}, {self.dz_plate:.4f}
        ]);
        topGeo.setAttribute('position', new THREE.BufferAttribute(topVerts, 3));
        topGeo.computeVertexNormals();
        scene.add(new THREE.Mesh(topGeo, plateMat));
        
        // Bottom Plate
        const botGeo = new THREE.BufferGeometry();
        const botVerts = new Float32Array([
            -subW/2, -subL/2, 0.0,
             subW/2, -subL/2, 0.0,
             subW/2, -subL/2 - {self.dy_plate:.4f}, {self.dz_plate:.4f},
            -subW/2, -subL/2, 0.0,
             subW/2, -subL/2 - {self.dy_plate:.4f}, {self.dz_plate:.4f},
            -subW/2, -subL/2 - {self.dy_plate:.4f}, {self.dz_plate:.4f}
        ]);
        botGeo.setAttribute('position', new THREE.BufferAttribute(botVerts, 3));
        botGeo.computeVertexNormals();
        scene.add(new THREE.Mesh(botGeo, plateMat));
        """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #f8fafc; }}
        #hud {{ position: absolute; top: 16px; left: 16px; background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 16px 20px; max-width: 380px; backdrop-filter: blur(8px); box-shadow: 0 4px 20px rgba(0,0,0,0.5); }}
        h1 {{ margin: 0 0 6px 0; font-size: 15px; font-weight: 600; color: #38bdf8; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #3b82f6; color: #fff; margin-bottom: 8px; }}
        p {{ margin: 4px 0; font-size: 12px; color: #cbd5e1; }}
        .metric {{ font-weight: 600; color: #f8fafc; }}
        .controls {{ margin-top: 10px; font-size: 11px; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <div class="badge">24 GHz Radar Array</div>
        <h1>{self.title}</h1>
        <p>Reference: <span class="metric">MDPI Electronics 2023, 12(7), 1512</span></p>
        <p>Frequency: <span class="metric">24.125 GHz (24.0 – 24.25 GHz)</span></p>
        <p>Substrate: <span class="metric">HF-350F (eps_r = 3.50, h = 0.254 mm)</span></p>
        <p>Configuration: <span class="metric">{plates_note}</span></p>
        <p>Expected Peak Gain: <span class="metric">{gain_str}</span></p>
        <p>Sidelobe Level: <span class="metric">{specs.get('proven_sidelobe_level_db', 'N/A')}</span></p>
        <div class="controls">
            Left Click + Drag: Rotate | Right Click + Drag: Pan | Scroll: Zoom
        </div>
    </div>

    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);
        
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(25, -45, 45);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lighting
        const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambLight);
        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight1.position.set(40, -30, 60);
        scene.add(dirLight1);
        const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
        dirLight2.position.set(-40, 30, -30);
        scene.add(dirLight2);
        
        const subW = {self.sub_w:.4f};
        const subL = {self.sub_l:.4f};
        const subH = {self.sub_h:.4f};
        
        // Substrate
        const subGeo = new THREE.BoxGeometry(subW, subL, subH);
        const subMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, transparent: true, opacity: 0.40 }});
        const sub = new THREE.Mesh(subGeo, subMat);
        sub.position.set(0, 0, subH / 2.0);
        scene.add(sub);
        
        // Ground plane
        const gndGeo = new THREE.BoxGeometry(subW, subL, 0.02);
        const gndMat = new THREE.MeshStandardMaterial({{ color: 0xb45309, metalness: 0.85, roughness: 0.25 }});
        const gnd = new THREE.Mesh(gndGeo, gndMat);
        gnd.position.set(0, 0, -0.01);
        scene.add(gnd);
        
        // Copper traces
        const traceMat = new THREE.MeshStandardMaterial({{ color: 0xd97706, metalness: 0.90, roughness: 0.20 }});
        const boxes = [
            {boxes_str}
        ];
        boxes.forEach(b => {{
            if (b.name.includes('disk')) return;
            const geo = new THREE.BoxGeometry(b.w, b.l, 0.035);
            const mesh = new THREE.Mesh(geo, traceMat);
            mesh.position.set(b.x, b.y, subH + 0.0175);
            scene.add(mesh);
        }});
        
        // Disk
        const diskGeo = new THREE.CylinderGeometry({self.r1:.4f}, {self.r1:.4f}, 0.035, 32);
        diskGeo.rotateX(Math.PI / 2);
        const diskMesh = new THREE.Mesh(diskGeo, traceMat);
        diskMesh.position.set(0.0, {self.y_port:.4f}, subH + 0.0175);
        scene.add(diskMesh);
        
        {plates_code}
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)



# ==============================================================================
# Model 6: Qorvo / DecaWave WB002 UWB Planar Monopole MAN Reference Antenna
# ==============================================================================

class QorvoWB002Antenna(BaseAntennaModel):
    """
    Qorvo / DecaWave WB002 Reference Planar Monopole Antenna ("MAN" Antenna).
    Reference: WB002 Hardware Build Instruction Version 1.1 (DecaWave / Qorvo)
    
    Stackup & Substrate:
      - Board Material: Standard FR-4 (eps_r = 4.30, tan_d = 0.020)
      - Substrate Thickness: 0.8 mm (31 mil)
      - Copper Thickness: 35 um (1 oz Cu, sigma = 5.8e7 S/m)
      - Board Dimensions: 31.0 mm x 40.0 mm
      - Operational Band: 3.0 GHz – 8.0 GHz (IEEE 802.15.4 UWB Channels 1–7)
      - Characteristic Feed: 50 Ohm Microstrip (Width = 1.625 mm)
      - Geometry: Exact Gerber-extracted 42-point "MAN" radiator and 26-point tapered ground plane
    """
    TOP_RADIATOR_PTS = [
        (-6.8968, 12.7978), (-5.7545, 12.5230), (-4.6062, 11.9168), (-3.5280, 11.1259),
        (-2.5930, 10.2930), (-1.8021, 9.4180), (-1.1542, 8.5006), (-0.6494, 7.5411),
        (-0.2878, 6.5405), (-0.0704, 5.4977), (0.0037, 4.4144), (0.0037, -4.4144),
        (-0.0704, -5.4977), (-0.2878, -6.5405), (-0.6494, -7.5411), (-1.1542, -8.5006),
        (-1.8021, -9.4180), (-2.5930, -10.2930), (-3.5280, -11.1259), (-4.6062, -11.9168),
        (-5.7545, -12.5230), (-6.8968, -12.7978), (-8.0334, -12.7416), (-9.1636, -12.3535),
        (-10.2868, -11.6351), (-11.4034, -10.5866), (-12.5123, -9.2074), (-13.6150, -7.4984),
        (-14.5767, -5.6241), (-15.2642, -3.7496), (-15.6766, -1.8749), (-15.8147, 0.0000),
        (-15.6766, 1.8749), (-15.2642, 3.7496), (-14.5767, 5.6241), (-13.6150, 7.4984),
        (-12.5123, 9.2074), (-11.4034, 10.5866), (-10.2868, 11.6351), (-9.1636, 12.3535),
        (-8.0334, 12.7416), (-6.8968, 12.7978)
    ]
    
    BOTTOM_GND_PTS = [
        (13.0127, 20.0139), (13.0127, -20.0139), (10.4501, -19.7756), (8.2136, -19.3213),
        (6.2898, -18.6506), (4.6777, -17.7615), (3.3787, -16.6557), (2.3919, -15.3324),
        (1.7188, -13.7919), (1.3584, -12.0360), (0.8244, -7.1714), (0.4394, -3.7610),
        (0.2663, -2.2921), (0.1263, 0.0000), (0.2663, 2.2921), (0.4394, 3.7610),
        (0.8244, 7.1714), (1.3584, 12.0360), (1.7188, 13.7919), (2.3919, 15.3324),
        (3.3787, 16.6557), (4.6777, 17.7615), (6.2898, 18.6506), (8.2136, 19.3213),
        (10.4501, 19.7756), (13.0127, 20.0139)
    ]

    def __init__(self, cells_per_wavelength: float = 60.0):
        super().__init__(
            name="qorvo_wb002",
            title="Qorvo WB002 UWB Planar Monopole MAN Reference Antenna (3 – 8 GHz)",
            doc_reference="Qorvo / DecaWave WB002 Hardware Build Instruction v1.1",
            f_center=5.50e9,
            f_start=3.00e9,
            f_stop=8.00e9,
            f_wide_start=2.50e9,
            f_wide_stop=8.50e9,
            cells_per_wavelength=max(60.0, float(cells_per_wavelength))
        )
        self.sub_xmin = -18.0000
        self.sub_xmax = 13.0127
        self.sub_ymin = -20.0139
        self.sub_ymax = 20.0139
        self.sub_w = self.sub_xmax - self.sub_xmin  # 31.0127 mm
        self.sub_l = self.sub_ymax - self.sub_ymin  # 40.0278 mm
        self.sub_h = 0.8000                         # mm (0.8 mm FR4)
        self.sub_eps_r = 4.30
        self.sub_tan_d = 0.0200
        self.sub_kappa = 2 * np.pi * self.f_center * EPS0 * self.sub_eps_r * self.sub_tan_d
        
        # 50-Ohm Feed Line
        self.feed_xmin = -5.0127
        self.feed_xmax = 13.0127
        self.feed_w = 1.6254
        self.feed_ymin = -self.feed_w / 2.0
        self.feed_ymax = self.feed_w / 2.0

    def get_proven_specs(self) -> dict:
        return {
            "source_doc": "Qorvo / DecaWave WB002 Hardware Build Instruction v1.1 (Figures 1–9 & Table 1)",
            "operating_band": "3.0 – 8.0 GHz Ultra-Wideband (IEEE 802.15.4 Channels 1–7)",
            "nominal_center_freq_ghz": 5.50,
            "proven_s11_reflection_db": "< -10.0 dB from 3.4 to >10 GHz (Resonance at ~4.25 GHz with S11 < -40 dB)",
            "proven_10db_bandwidth_mhz": "> 4500 MHz (3.4 – >8.0 GHz, > 80% fractional bandwidth)",
            "proven_peak_gain_dbi": "2.2 dBi at 4.0 GHz, 3.3 dBi at 6.5 GHz (Omnidirectional)",
            "proven_elevation_hpbw_deg": "Donut-shaped monopole elevation cut (Figure 3 & 4)",
            "proven_azimuth_hpbw_deg": "Omnidirectional in Azimuth plane (Theta=90°)",
            "fidelity_factor": "> 90% across 3 – 8 GHz in azimuth plane",
            "group_delay_variation": "< ±100 ps across antenna rotation",
            "antenna_dimensions": f"{self.sub_w:.1f} mm x {self.sub_l:.1f} mm x {self.sub_h:.1f} mm compact PCB",
            "substrate_material": "Standard FR-4 (eps_r = 4.30, tan_d = 0.020, h = 0.80 mm / 31 mil)"
        }

    def create_simulation(self, end_criteria: float = 1e-6, nr_ts: int = 500000, active_channels=1):
        """Construct CSXCAD model and openEMS FDTD configuration for WB002."""
        FDTD = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
        fc = 3.0e9  # 3.0 GHz pulse bandwidth around 5.5 GHz (covering 2.5 to 8.5 GHz)
        FDTD.SetGaussExcite(self.f_center, fc)
        FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(self.unit)
        
        # 1. Materials
        fr4 = CSX.AddMaterial('FR4', epsilon=self.sub_eps_r, kappa=self.sub_kappa)
        copper = CSX.AddMetal('Copper')
        
        # 2. Substrate
        fr4.AddBox([self.sub_xmin, self.sub_ymin, 0.0],
                   [self.sub_xmax, self.sub_ymax, self.sub_h], priority=1)
                   
        # 3. Bottom Layer: Ground Plane with curved taper
        gnd_x = [pt[0] for pt in self.BOTTOM_GND_PTS]
        gnd_y = [pt[1] for pt in self.BOTTOM_GND_PTS]
        copper.AddLinPoly(np.array([gnd_x, gnd_y]), 'z', 0.0, 0.0, priority=10)
        
        # 4. Top Layer: 50-Ohm Feed Line Box
        copper.AddBox([self.feed_xmin, self.feed_ymin, self.sub_h],
                      [self.feed_xmax, self.feed_ymax, self.sub_h], priority=10)
                      
        # 5. Top Layer: MAN Radiator Polygon
        rad_x = [pt[0] for pt in self.TOP_RADIATOR_PTS]
        rad_y = [pt[1] for pt in self.TOP_RADIATOR_PTS]
        copper.AddLinPoly(np.array([rad_x, rad_y]), 'z', self.sub_h, 0.0, priority=10)
        
        # 6. Lumped 50-Ohm Port at board edge (x = 13.0127)
        port_start = [self.feed_xmax - 0.4, self.feed_ymin, 0.0]
        port_stop = [self.feed_xmax, self.feed_ymax, self.sub_h]
        port = FDTD.AddLumpedPort(1, 50.0, port_start, port_stop, 'z', 1.0, priority=20, edges2grid='xy')
        
        # 7. Mesh Definition (guaranteeing >= 60 cells/lambda everywhere)
        res = min(self.lambda0_mm / self.cells_per_wavelength, 0.50)  # <= 0.50 mm
        air_pad = self.lambda0_mm * 0.45
        
        # Z lines (exact mesh lines at z=0.0 and z=0.8)
        nz_sub = 4
        z_sub = np.linspace(0.0, self.sub_h, nz_sub + 1)
        z_raw = np.concatenate([[-air_pad], z_sub, [self.sub_h + air_pad]])
        z_lines = self._smooth_mesh(z_raw, res, ratio=1.30)
        
        # X lines
        x_features = [
            self.sub_xmin - air_pad, self.sub_xmin, -15.8147, -6.8968, -5.0127,
            0.0, 0.1263, 0.8244, 10.4501, self.sub_xmax, self.sub_xmax + air_pad
        ]
        x_lines = self._smooth_mesh(np.array(x_features), res, ratio=1.30)
        
        # Y lines
        y_features = [
            self.sub_ymin - air_pad, self.sub_ymin, -12.7978, -4.4144,
            self.feed_ymin, 0.0, self.feed_ymax, 4.4144, 12.7978,
            self.sub_ymax, self.sub_ymax + air_pad
        ]
        y_lines = self._smooth_mesh(np.array(y_features), res, ratio=1.30)
        
        mesh.AddLine('x', x_lines)
        mesh.AddLine('y', y_lines)
        mesh.AddLine('z', z_lines)
        
        # 8. NF2FF
        nf2ff = FDTD.CreateNF2FFBox()
        return FDTD, CSX, [port], nf2ff

    def draw_layout(self, output_path: str):
        """Generate high-resolution 2D engineering layout diagram."""
        fig, (ax_plan, ax_side) = plt.subplots(1, 2, figsize=(15, 7),
                                               gridspec_kw={'width_ratios': [1.3, 0.9]},
                                               tight_layout=True)
        # Plan view
        sub_rect = mpatches.Rectangle((self.sub_xmin, self.sub_ymin), self.sub_w, self.sub_l,
                                     facecolor='#e2e8f0', edgecolor='#64748b', linewidth=1.5,
                                     label=f'FR-4 Substrate ({self.sub_w:.1f}x{self.sub_l:.1f} mm, h=0.8mm)')
        ax_plan.add_patch(sub_rect)
        
        # Bottom Ground Plane
        gnd_poly = mpatches.Polygon(self.BOTTOM_GND_PTS, facecolor='#b45309', edgecolor='#78350f',
                                    alpha=0.45, linewidth=1.2, label='Bottom Ground Plane (with curved taper)')
        ax_plan.add_patch(gnd_poly)
        
        # Top Feed Line
        feed_rect = mpatches.Rectangle((self.feed_xmin, self.feed_ymin), self.feed_xmax - self.feed_xmin, self.feed_w,
                                      facecolor='#ea580c', edgecolor='#9a3412', linewidth=1.0,
                                      label=f'Top 50-Ohm Feed Line (W={self.feed_w:.2f} mm)')
        ax_plan.add_patch(feed_rect)
        
        # Top MAN Radiator
        rad_poly = mpatches.Polygon(self.TOP_RADIATOR_PTS, facecolor='#dc2626', edgecolor='#991b1b',
                                    alpha=0.90, linewidth=1.2, label='Top MAN Radiator (3 – 8 GHz)')
        ax_plan.add_patch(rad_poly)
        
        # Annotations
        ax_plan.annotate(f'Board Width = {self.sub_w:.1f} mm', xy=((self.sub_xmin + self.sub_xmax)/2, self.sub_ymax + 1.2),
                         ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e293b')
        ax_plan.annotate(f'Board Length = {self.sub_l:.1f} mm', xy=(self.sub_xmin - 1.2, 0),
                         ha='right', va='center', fontsize=9.5, fontweight='bold', color='#1e293b', rotation=90)
        ax_plan.annotate('Port 1 (50-Ohm Feed)', xy=(self.sub_xmax, 0),
                         xytext=(self.sub_xmax + 2.5, 0),
                         arrowprops=dict(facecolor='#ea580c', shrink=0.08, width=1.5, headwidth=6),
                         ha='left', va='center', fontsize=9, fontweight='bold', color='#c2410c')
        ax_plan.annotate('Curved Ground Taper', xy=(0.126, 0.0),
                         xytext=(4.0, -8.0),
                         arrowprops=dict(facecolor='#b45309', shrink=0.08, width=1.0, headwidth=5),
                         fontsize=8.5, color='#b45309')
                         
        ax_plan.set_xlim(self.sub_xmin - 4.0, self.sub_xmax + 12.0)
        ax_plan.set_ylim(self.sub_ymin - 3.0, self.sub_ymax + 3.0)
        ax_plan.set_aspect('equal')
        ax_plan.set_title(f"{self.title}\n(Exact Gerber Layout from {self.doc_reference})", fontsize=11, fontweight='bold')
        ax_plan.set_xlabel('X Dimension (mm)')
        ax_plan.set_ylabel('Y Dimension (mm)')
        ax_plan.legend(loc='lower left', fontsize=8.5)
        ax_plan.grid(True, linestyle=':', alpha=0.5)
        
        # Side view (Stackup)
        ax_side.set_title("PCB Stackup Cross-Section (XZ)", fontsize=11, fontweight='bold')
        ax_side.add_patch(mpatches.Rectangle((self.sub_xmin, 0.0), self.sub_w, self.sub_h,
                                            facecolor='#e2e8f0', edgecolor='#64748b', linewidth=1.2,
                                            label=f'FR-4 Core ({self.sub_h*1000:.0f} um / 31 mil, eps_r=4.30)'))
        ax_side.plot([self.BOTTOM_GND_PTS[13][0], self.sub_xmax], [0.0, 0.0],
                     color='#b45309', linewidth=3.0, label='Bottom Cu Ground (35 um)')
        ax_side.plot([self.feed_xmin, self.feed_xmax], [self.sub_h, self.sub_h],
                     color='#ea580c', linewidth=2.5, label='Top Cu 50-Ohm Feed (35 um)')
        ax_side.plot([min(p[0] for p in self.TOP_RADIATOR_PTS), max(p[0] for p in self.TOP_RADIATOR_PTS)],
                     [self.sub_h, self.sub_h], color='#dc2626', linewidth=3.0, label='Top Cu MAN Radiator (35 um)')
                     
        ax_side.plot([self.feed_xmax, self.feed_xmax], [0.0, self.sub_h], color='#c2410c', linewidth=2.5, linestyle='--')
        ax_side.annotate('50-Ohm SMA Port', xy=(self.feed_xmax, self.sub_h / 2.0),
                         xytext=(self.feed_xmax + 2.0, self.sub_h / 2.0),
                         ha='left', va='center', fontsize=8.5, fontweight='bold', color='#c2410c')
                         
        ax_side.set_xlim(self.sub_xmin - 3.0, self.sub_xmax + 10.0)
        ax_side.set_ylim(-1.5, 3.5)
        ax_side.set_xlabel('X Dimension (mm)')
        ax_side.set_ylabel('Z Dimension (mm)')
        ax_side.legend(loc='upper left', fontsize=8.5)
        ax_side.grid(True, linestyle=':', alpha=0.5)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def draw_3d_view(self, output_path: str):
        """Generate publication-ready 3D isometric perspective diagram."""
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Substrate
        z_top = self.sub_h
        xx, yy = np.meshgrid([self.sub_xmin, self.sub_xmax], [self.sub_ymin, self.sub_ymax])
        ax.plot_surface(xx, yy, np.full_like(xx, z_top), color='#94a3b8', alpha=0.30)
        ax.plot_surface(xx, yy, np.zeros_like(xx), color='#94a3b8', alpha=0.15)
        
        # Bottom Ground Plane polygon
        gnd_3d = [[(pt[0], pt[1], 0.0) for pt in self.BOTTOM_GND_PTS]]
        ax.add_collection3d(Poly3DCollection(gnd_3d, facecolors='#b45309', edgecolors='#78350f', alpha=0.7, linewidth=0.8))
        
        # Top Feed Line Box
        feed_3d = [[
            (self.feed_xmin, self.feed_ymin, z_top + 0.02),
            (self.feed_xmax, self.feed_ymin, z_top + 0.02),
            (self.feed_xmax, self.feed_ymax, z_top + 0.02),
            (self.feed_xmin, self.feed_ymax, z_top + 0.02)
        ]]
        ax.add_collection3d(Poly3DCollection(feed_3d, facecolors='#ea580c', edgecolors='#9a3412', alpha=0.9, linewidth=0.8))
        
        # Top Radiator polygon
        rad_3d = [[(pt[0], pt[1], z_top + 0.02) for pt in self.TOP_RADIATOR_PTS]]
        ax.add_collection3d(Poly3DCollection(rad_3d, facecolors='#dc2626', edgecolors='#991b1b', alpha=0.9, linewidth=0.8))
        
        # Port marker
        ax.plot([self.feed_xmax, self.feed_xmax], [0, 0], [0, z_top], color='#ea580c', linewidth=3)
        ax.text(self.feed_xmax + 1.0, 0, z_top / 2.0, 'P1', color='#ea580c', fontweight='bold', fontsize=9)
        
        ax.set_xlim(self.sub_xmin - 3, self.sub_xmax + 3)
        ax.set_ylim(self.sub_ymin - 3, self.sub_ymax + 3)
        ax.set_zlim(-1.5, 3.5)
        ax.set_title(f"{self.title}\n3D Isometric Structural Model", fontsize=11, fontweight='bold', pad=15)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.view_init(elev=28, azim=-55)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def draw_interactive_3d(self, output_path: str):
        """Generate interactive 3D WebGL Three.js HTML model with dark-theme HUD."""
        rad_js = ", ".join([f"[{pt[0]:.4f}, {pt[1]:.4f}]" for pt in self.TOP_RADIATOR_PTS])
        gnd_js = ", ".join([f"[{pt[0]:.4f}, {pt[1]:.4f}]" for pt in self.BOTTOM_GND_PTS])
        specs = self.get_proven_specs()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #f8fafc; }}
        #hud {{ position: absolute; top: 16px; left: 16px; background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 16px 20px; max-width: 380px; backdrop-filter: blur(8px); box-shadow: 0 4px 20px rgba(0,0,0,0.5); }}
        h1 {{ margin: 0 0 6px 0; font-size: 15px; font-weight: 600; color: #38bdf8; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #3b82f6; color: #fff; margin-bottom: 8px; }}
        p {{ margin: 4px 0; font-size: 12px; color: #cbd5e1; }}
        .metric {{ font-weight: 600; color: #f8fafc; }}
        .controls {{ margin-top: 10px; font-size: 11px; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <div class="badge">UWB Monopole Reference</div>
        <h1>{self.title}</h1>
        <p>Reference: <span class="metric">Qorvo / DecaWave WB002 Build Instruction</span></p>
        <p>Operational Band: <span class="metric">3.0 – 8.0 GHz (IEEE 802.15.4 UWB)</span></p>
        <p>Dimensions: <span class="metric">{self.sub_w:.1f} x {self.sub_l:.1f} x {self.sub_h:.1f} mm</span></p>
        <p>Substrate: <span class="metric">FR-4 (eps_r = 4.30, h = 0.8 mm)</span></p>
        <p>Peak Gain: <span class="metric">2.2 dBi @ 4.0 GHz, 3.3 dBi @ 6.5 GHz</span></p>
        <p>Radiation Pattern: <span class="metric">Omnidirectional (Azimuth plane)</span></p>
        <p>Group Delay Var: <span class="metric">&lt; ±100 ps across rotation</span></p>
        <div class="controls">
            Left Click + Drag: Rotate | Right Click + Drag: Pan | Scroll: Zoom
        </div>
    </div>

    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);
        
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, -60, 45);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lighting
        const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambLight);
        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight1.position.set(30, -40, 50);
        scene.add(dirLight1);
        const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
        dirLight2.position.set(-30, 40, -30);
        scene.add(dirLight2);
        
        const subW = {self.sub_w:.4f};
        const subL = {self.sub_l:.4f};
        const subH = {self.sub_h:.4f};
        const subX0 = {self.sub_xmin:.4f};
        const subY0 = {self.sub_ymin:.4f};
        
        // Substrate Box
        const subGeo = new THREE.BoxGeometry(subW, subL, subH);
        const subMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, transparent: true, opacity: 0.35 }});
        const sub = new THREE.Mesh(subGeo, subMat);
        sub.position.set(subX0 + subW / 2.0, subY0 + subL / 2.0, subH / 2.0);
        scene.add(sub);
        
        // Materials
        const gndMat = new THREE.MeshStandardMaterial({{ color: 0xb45309, metalness: 0.85, roughness: 0.25, side: THREE.DoubleSide }});
        const feedMat = new THREE.MeshStandardMaterial({{ color: 0xea580c, metalness: 0.90, roughness: 0.20 }});
        const radMat = new THREE.MeshStandardMaterial({{ color: 0xdc2626, metalness: 0.90, roughness: 0.20, side: THREE.DoubleSide }});
        
        // Bottom Ground Shape
        const gndPts = [{gnd_js}];
        const gndShape = new THREE.Shape();
        gndShape.moveTo(gndPts[0][0], gndPts[0][1]);
        for (let i = 1; i < gndPts.length; i++) {{
            gndShape.lineTo(gndPts[i][0], gndPts[i][1]);
        }}
        const gndGeo = new THREE.ShapeGeometry(gndShape);
        const gndMesh = new THREE.Mesh(gndGeo, gndMat);
        gndMesh.position.set(0, 0, -0.01);
        scene.add(gndMesh);
        
        // Top 50-Ohm Feed Line Box
        const feedLen = {self.feed_xmax - self.feed_xmin:.4f};
        const feedW = {self.feed_w:.4f};
        const feedGeo = new THREE.BoxGeometry(feedLen, feedW, 0.035);
        const feedMesh = new THREE.Mesh(feedGeo, feedMat);
        feedMesh.position.set({(self.feed_xmin + self.feed_xmax)/2.0:.4f}, 0, subH + 0.0175);
        scene.add(feedMesh);
        
        // Top MAN Radiator Shape
        const radPts = [{rad_js}];
        const radShape = new THREE.Shape();
        radShape.moveTo(radPts[0][0], radPts[0][1]);
        for (let i = 1; i < radPts.length; i++) {{
            radShape.lineTo(radPts[i][0], radPts[i][1]);
        }}
        const radGeo = new THREE.ShapeGeometry(radShape);
        const radMesh = new THREE.Mesh(radGeo, radMat);
        radMesh.position.set(0, 0, subH + 0.01);
        scene.add(radMesh);
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


# ==============================================================================
# Model Catalog & Factory
# ==============================================================================

SUPPORTED_ANTENNAS = {
    'ti_77ghz_rx': TI77GHzRXAntenna,
    'ti_24ghz_an043': TI24GHzAN043Antenna,
    'ti_24ghz_swru120d': TI24GHzSWRU120DAntenna,
    'mdpi_24ghz_array': lambda **kwargs: MDPI24GHzPatchArrayAntenna(with_plates=False, **kwargs),
    'mdpi_24ghz_plates': lambda **kwargs: MDPI24GHzPatchArrayAntenna(with_plates=True, **kwargs),
    'qorvo_wb002': QorvoWB002Antenna,
}

def get_antenna_model(antenna_name: str = 'ti_77ghz_rx', **kwargs) -> BaseAntennaModel:
    """Factory function to instantiate a proven antenna model."""
    name_clean = str(antenna_name).strip().lower()
    if name_clean in SUPPORTED_ANTENNAS:
        return SUPPORTED_ANTENNAS[name_clean](**kwargs)
    
    # Aliases
    if 'wb002' in name_clean or 'man' in name_clean or 'qorvo' in name_clean or 'decawave' in name_clean:
        return QorvoWB002Antenna(**kwargs)
    elif 'plates' in name_clean:
        return MDPI24GHzPatchArrayAntenna(with_plates=True, **kwargs)
    elif '1512' in name_clean or 'mdpi' in name_clean:
        return MDPI24GHzPatchArrayAntenna(with_plates=False, **kwargs)
    elif '77' in name_clean or 'swru553' in name_clean:
        return TI77GHzRXAntenna(**kwargs)
    elif 'an043' in name_clean or 'mifa' in name_clean:
        return TI24GHzAN043Antenna(**kwargs)
    elif 'swru120' in name_clean or 'dn007' in name_clean or 'ifa' in name_clean:
        return TI24GHzSWRU120DAntenna(**kwargs)
    else:
        raise ValueError(f"Unknown antenna '{antenna_name}'. Supported antennas: {list(SUPPORTED_ANTENNAS.keys())}")


