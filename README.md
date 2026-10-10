# 77 GHz 4-Channel Patch Antenna Array (openEMS FDTD Simulation)

A high-performance full-wave electromagnetic simulation suite for a **77 GHz 4-channel series-fed microstrip patch antenna array** operating across the **76 GHz to 78 GHz** automotive radar band. The model is built on an automotive-grade **4-layer PCB stackup** and simulated using **openEMS** (Finite-Difference Time-Domain / FDTD) with GPU hardware acceleration (Vulkan) on Windows.

Every simulation run records its exact start time and outputs all figures, multi-port S-parameters, mutual coupling curves, far-field radiation patterns, interactive 3D WebGL models, and CSXCAD geometry files into a dedicated timestamped subfolder inside `SimResult/` (e.g., `SimResult/YYYY-MM-DD_HH-MM-SS/`).

---

## 1. Array Architecture & PCB Stackup

### Multi-Channel Spatial Layout
The antenna features **4 parallel series-fed linear patch arrays** arranged side-by-side along the transverse $X$-axis with **$5.0\text{ mm}$ center-to-center spacing** ($\approx 1.28\,\lambda_0$ at $77\text{ GHz}$).

- **Number of Channels**: 4 independent channels
- **Channel Offsets ($X$-axis)**: `[-7.5, -2.5, +2.5, +7.5] mm`
- **Port Excitation**: Port 1 driven with an excitation Gaussian pulse; Ports 2, 3, and 4 terminated in $50\,\Omega$ matched loads for rigorous mutual coupling extraction ($S_{21}, S_{31}, S_{41}$).
- **Overall Substrate Dimensions**: $22.35\text{ mm} \times 10.51\text{ mm}$ ($W \times L$)

### 4-Layer Automotive mmWave Stackup
| Layer | Material / Type | Thickness | Electrical Properties | Function |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1 (Top Cu)** | Annealed Copper | $35\ \mu\text{m}$ (1 oz) | $\sigma = 5.8 \times 10^7\text{ S/m}$ | 4 Series Arrays & $50\,\Omega$ feeds |
| **Dielectric 1** | **Rogers RO3003** | **$0.127\text{ mm}$ (5 mil)** | $\varepsilon_r = 3.00,\ \tan\delta = 0.0010$ | Low-loss mmWave RF cavity |
| **Layer 2 (RF Ground)**| Continuous Copper Plane | $35\ \mu\text{m}$ (1 oz) | Reference plane ($z = 0$) | Microstrip ground return & RF barrier |
| **Dielectric 2** | **FR-4 Core** | **$0.800\text{ mm}$ (31.5 mil)** | $\varepsilon_r = 4.40,\ \tan\delta = 0.020$ | Mechanical rigidity & baseboard |
| **Layer 4 (Bottom Cu)**| Continuous Copper Plane | $35\ \mu\text{m}$ (1 oz) | Backside shield ($z = -0.80\text{ mm}$) | Baseboard shielding & heat sinking |
| **Total Board** | Composite Stackup | **$\approx 0.93\text{ mm}$** | Rigid automotive-grade PCB | Rugged mmWave radar sensor |

---

## 2. RF Performance Specifications & Measured Metrics

| Parameter | Simulated Value @ 77.0 GHz | Target Specification | Status |
| :--- | :--- | :--- | :--- |
| **Resonant Frequency ($f_{\text{res}}$)** | **$77.020\text{ GHz}$** | $77.000\text{ GHz}$ | Perfectly centered ($\Delta f = 20\text{ MHz}$) |
| **Port 1 Return Loss ($S_{11}$)** | **$-22.07\text{ dB}$** ($VSWR = 1.17$) | $<-10.0\text{ dB}$ | Excellent matching |
| **Input Impedance ($Z_{\text{in}}$)** | **$43.71 - 3.73j\ \Omega$** | $50.0\ \Omega$ | Matched within $6.3\ \Omega$ |
| **-10 dB Bandwidth** | **$0.930\text{ GHz}$** ($76.53 - 77.46\text{ GHz}$) | $>0.50\text{ GHz}$ | Full automotive FMCW sub-band |
| **Adjacent Coupling ($S_{21}$)** | **$-39.38\text{ dB}$** ($5.0\text{ mm}$ spacing) | $<-25.0\text{ dB}$ | High channel isolation |
| **Cross-Coupling ($S_{31}$)** | **$-64.47\text{ dB}$** ($10.0\text{ mm}$ spacing) | $<-30.0\text{ dB}$ | Negligible crosstalk |
| **Far Coupling ($S_{41}$)** | **$-46.23\text{ dB}$** ($15.0\text{ mm}$ spacing) | $<-30.0\text{ dB}$ | Negligible crosstalk |
| **Single-Channel Directivity ($D_{\text{max}}$)** | **$10.25\text{ dBi}$** | $>10.0\text{ dBi}$ | Elevation focused |
| **Elevation HPBW (E-plane)** | **$20.6^\circ$** | $<25^\circ$ | High vertical resolution |
| **Azimuth HPBW (Single Channel)** | **$154.6^\circ$** | Broad coverage | Wide FOV per channel |
| **4-Ch Array Directivity ($D_{\text{array}}$)** | **$15.92\text{ dBi}$** ($+5.67\text{ dB}$ array gain) | $>15.0\text{ dBi}$ | Synthesized broadside beam |
| **4-Ch Array Azimuth HPBW** | **$15.1^\circ$** | $<20^\circ$ | High azimuthal angular resolution |

---

## 3. Project Directory Structure

```
Test_Antenna/
├── openEMS/                       # Bundled openEMS + CSXCAD Windows binaries & wheels
│   ├── openEMS.exe               # FDTD solver executable
│   ├── AppCSXCAD.exe             # 3D CAD visualization executable
│   └── python/                   # Pre-compiled Python 3.13 / 3.14 wheels
├── .idea/                         # PyCharm project settings and run configuration
│   └── runConfigurations/
│       └── Simulate_Antenna_77GHz.xml
├── .venv/                         # Project virtual environment (Python 3.13)
├── SimResult/                     # All simulation runs go here!
│   └── YYYY-MM-DD_HH-MM-SS/      # Timestamped simulation run folder
│       ├── antenna_layout.png     # 2D schematic showing 4 channels & 4-layer stackup
│       ├── antenna_3d_view.png     # 3D isometric structural plot
│       ├── antenna_3d_interactive.html # Interactive 3D WebGL model (orbit/zoom/pan)
│       ├── s11_reflection.png     # S11 return loss plot (76 - 78 GHz)
│       ├── mutual_coupling.png    # Inter-channel mutual coupling (S21, S31, S41)
│       ├── input_impedance.png    # Complex Zin (Resistance + Reactance)
│       ├── radiation_pattern_polar.png # Polar patterns (single channel vs 4-ch array)
│       ├── radiation_pattern_cartesian.png # Cartesian directivity vs elevation angle
│       ├── s_parameters_data.csv  # S11, S21, S31, S41, Zin numerical data
│       ├── radiation_pattern_77ghz.csv # Numerical far-field directivity table
│       ├── openEMS_model.xml      # CSXCAD 3D geometry and mesh file
│       ├── openEMS.log            # Full raw openEMS FDTD solver engine log
│       ├── simulation.log         # Complete Python execution log
│       ├── simulation_summary.json# Machine-readable metrics summary
│       └── simulation_report.txt  # Human-readable formatted text report
├── antenna_model.py               # 4-Channel array geometry, 4-layer stackup & mesh
├── simulate.py                    # Multi-port FDTD simulation and post-processing engine
├── main.py                        # CLI & PyCharm entry point
├── openems_config.py              # Dynamic DLL and environment configuration
├── requirements.txt               # Python package dependencies
└── README.md                      # Documentation
```

---

## 4. Running the Simulation

### Option A: In PyCharm
1. Open the project folder `Test_Antenna` in PyCharm.
2. Ensure the Python interpreter is set to `.venv` (Python 3.13).
3. Select the run configuration **`Simulate Antenna 77GHz`** from the run dropdown.
4. Click **Run ▶** (or press `Shift + F10`).

### Option B: Terminal / PowerShell
```powershell
# Run with GPU acceleration (default: Channel 1 active, 50,000 max timesteps)
.\.venv\Scripts\python main.py

# Run on a specific channel (e.g. Channel 2, 3, or 4) with others 50-Ohm terminated
.\.venv\Scripts\python main.py --channel 2

# Run with all 4 channels driven simultaneously (active broadside beamforming array)
.\.venv\Scripts\python main.py --channel all

# Run with custom max timesteps (default: 50000, eliminating openEMS warning)
.\.venv\Scripts\python main.py --max-timesteps 60000

# Optional: Run with CPU multithreading
.\.venv\Scripts\python main.py --engine multithreaded

# Optional: View the 3D model and mesh in AppCSXCAD viewer
.\.venv\Scripts\python main.py --view

# Optional: Re-run post-processing on an existing simulation folder
.\.venv\Scripts\python main.py --post-process SimResult\2026-10-09_19-55-03 --channel all
```

---

## 5. Deliverables & Outputs

Each simulation run generates a self-contained results folder in `SimResult/` with:

1. **`antenna_layout.png`**: Dimensioned 2D engineering layout featuring Top View (XY) of all 4 channels ($5.0\text{ mm}$ spacing) and Side Elevation View (YZ) showing the 4-layer stackup.
2. **`antenna_3d_view.png`**: Publication-ready 3D isometric perspective plot displaying all copper traces, ground planes, Rogers RO3003 RF cavity, and FR-4 core.
3. **`antenna_3d_interactive.html`**: Interactive Three.js 3D model openable in any modern web browser to rotate, pan, and inspect the array geometry.
4. **`s11_reflection.png`**: Return loss curve ($S_{11}$) across the 76–78 GHz automotive band highlighting resonant frequency and -10 dB bandwidth.
5. **`mutual_coupling.png`**: Inter-channel mutual coupling isolation curves ($S_{21}, S_{31}, S_{41}$) across the band.
6. **`input_impedance.png`**: Complex input impedance $Z_{\text{in}} = R + jX$ versus frequency.
7. **`radiation_pattern_polar.png`**: 2D polar directivity patterns comparing a single active channel with the 4-channel synthesized broadside beamforming array.
8. **`radiation_pattern_cartesian.png`**: Cartesian radiation pattern with half-power beamwidth (HPBW) markers.
9. **`s_parameters_data.csv`**: Full multi-port S-parameter matrix and impedance data.
10. **`radiation_pattern_77ghz.csv`**: Angular far-field directivity data for E-plane and H-plane cuts.
11. **`openEMS_model.xml`**: CSXCAD 3D geometry file inspectable with `AppCSXCAD.exe`.
12. **`openEMS.log`**: Raw openEMS FDTD solver engine log capturing full GPU Vulkan initialization, operator matrix allocation, UPML layer counts, timestep iterations, speed (MC/s), and energy decay.
13. **`simulation.log`**: Complete end-to-end Python execution log recording configuration parameters, mesh statistics, solver progress, and RF extraction steps.
14. **`simulation_summary.json`**: Machine-readable JSON summary containing all key metrics.
15. **`simulation_report.txt`**: Formatted text report summarizing array architecture and RF performance.
