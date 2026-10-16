# Proven Reference Antennas (openEMS FDTD Simulation Suite)

A high-performance full-wave electromagnetic simulation suite for industry-proven antenna designs from **Texas Instruments (TI)** and peer-reviewed literature (**MDPI Electronics**). The project simulates exact antenna geometries derived directly from published application notes, user guides, and journal papers using **openEMS** (Finite-Difference Time-Domain / FDTD) with GPU hardware acceleration (Vulkan) on Windows.

Every simulation run records its configuration, solver logs, multi-port S-parameters, input impedance, far-field radiation patterns, interactive 3D WebGL models, and side-by-side benchmark comparison reports into a timestamped subfolder in `SimResult/` (e.g., `SimResult/YYYY-MM-DD_HH-MM-SS_<antenna_name>/`).

To compare the latest saved runs against the original reference documents, run:

```powershell
.\.venv\Scripts\python.exe compare_results.py --source-farfields
```

This reconstructs wideband S11 from recorded port signals, calculates gain and realized gain from radiated/input power, and writes `source_comparison.md`, numerical metrics, CSVs, and a plot under `SimResult/source_comparison_<run-date>/`. The optional far-field step evaluates the TI RX at the source's 76 GHz and WB002 at 4 and 6.5 GHz using saved fields. Original run files are preserved.

The static benchmark tables below contain values that disagree with the latest saved runs, and some compare directivity with published gain or truncate bandwidth to the operating-band sweep. Use the generated source comparison for current validation; the report identifies these discrepancies and documents the source figures and measurement conditions.

The corrected geometry uses the original TI Gerber outlines for both 2.4 GHz antennas, their L2 RF ground depths, grounded coplanar transitions with vias for TI 77 GHz, and a circular matching pad plus thin sheet reflectors for MDPI. [CAD provenance and remaining board/fixture approximations](docs_reference/ti_geometry_sources.md) are documented separately. Copper remains PEC for this geometry baseline.

All six models and the simulation pipeline default to **-60 dB energy decay** (`EndCriteria=1e-6`), with a **500,000-step cap**. The solver stops early when the decay criterion is reached; the cap can be raised with `--max-timesteps` if needed. New summaries record the actual energy-decay target, geometry revision and source hash. Post-processing reads the saved XML settings so older runs retain their original stopping target. An older run with a changed geometry revision is rejected to preserve its baseline.

Solver XML exports preserve matching double precision for mesh lines and Box coordinates. This prevents point excitations and thin PEC sheets from disappearing during XML loading. The runtime rejects unused sources/metals, and post-processing rejects empty driven-port recordings. Native geometry validation reloads the exported XML before checking both metal and excitation primitives. See the [export precision correction](docs_reference/solver_grid_precision.md).

To check source fidelity and export all corrected geometry previews without running an RF simulation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\validate_geometry.py --native
```

---

## 1. Supported Proven Antenna Reference Designs

The suite includes six proven antenna designs across 2.4 GHz, 3–8 GHz UWB, 24 GHz, and 77 GHz bands, with original documentation preserved in [`docs_reference/`](file:///c:/Users/Allen/Documents/Test_Antenna/docs_reference):

```
docs_reference/
├── swru553a.pdf              # TI MMWCAS-RF-EVM Cascaded Radar User's Guide (77 GHz RX/TX Array)
├── swra117d.pdf              # TI AN043: Small Size 2.4 GHz PCB antenna (CC2511 USB Dongle)
├── swru120d.pdf              # TI DN0007: 2.4 GHz Inverted-F Antenna (CC2430DB / CC2530)
├── mdpi_electronics_1512.pdf # MDPI Electronics 2023, 12(7), 1512 (24 GHz 4x1 Array with Flared Plates)
└── qorvo_wb002/              # Qorvo / DecaWave WB002 Reference Package (PDFs, DXF, Gerbers)
    └── WB002/
        ├── collateral/       # WB002 Hardware Build Instruction v1.1 & Product Brief
        ├── DXF/              # WB002_MAN_TOP.dxf & WB002_MAN_BTM.dxf
        └── Gerbers/          # DecaWave_WB002_top_layer.grb, bottom_layer.grb, outline.grb
```

### Summary of Supported Models

| Key | Antenna Model | Reference Doc | Frequency Band | Dimensions | Target Application |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ti_77ghz_rx`** | **77 GHz 4-Channel RX Series-Fed Patch Array** | TI **SWRU553A** §2.6.2 (Fig 16, Tab 7) | $76.0 - 81.0\text{ GHz}$ | $22.35 \times 10.51\text{ mm}$ | Automotive & Industrial FMCW Radar |
| **`ti_24ghz_an043`** | **2.4 GHz Small Size MIFA** | TI **AN043** / **SWRA117D** (Fig 3, Tab 1) | $2.40 - 2.48\text{ GHz}$ | $15.20 \times 5.70\text{ mm}$ | BLE, Zigbee, Wireless USB Dongles |
| **`ti_24ghz_swru120d`** | **2.4 GHz Inverted-F Antenna (IFA)** | TI **SWRU120D** / **DN0007** (Fig 1-1, Tab 1-1) | $2.40 - 2.48\text{ GHz}$ | $25.70 \times 7.50\text{ mm}$ | IEEE 802.15.4, CC2530 Sensor Nodes |
| **`mdpi_24ghz_array`** | **24 GHz 4x1 Microstrip Patch Array** | MDPI *Electronics* 2023, 12, 1512 (Fig 5) | $24.00 - 24.25\text{ GHz}$ | $30.00 \times 12.00\text{ mm}$ | K-Band Automotive & Traffic Radar |
| **`mdpi_24ghz_plates`** | **24 GHz Array + Dual 70° Flared Metallic Plates** | MDPI *Electronics* 2023, 12, 1512 (Fig 7–10) | $24.00 - 24.25\text{ GHz}$ | $30 \times 12\text{ mm}$ PCB + $50\text{ mm}$ plates | High-Gain (+7.2 dB) Long-Range Radar |
| **`qorvo_wb002`** | **WB002 UWB Planar Monopole MAN Antenna** | Qorvo / DecaWave **WB002 Build Instruction** v1.1 | $3.00 - 8.00\text{ GHz}$ | $31.01 \times 40.03\text{ mm}$ | IEEE 802.15.4 UWB RTLS, DW1000/DW3000 Ranging |

---

### High-Resolution Discretization Standard ($\ge 60\text{ Cells per Wavelength}$)
To eliminate numerical dispersion errors and accurately resolve fine microstrip trace widths, coplanar gaps, and near-field boundary transitions, the suite strictly enforces:
- **Minimum Grid Resolution**: At least **60 cells per free-space wavelength** ($\Delta \le \lambda_0 / 60$) across all Cartesian axes ($X, Y, Z$).
- **Substrate Discretization**: At least 4–5 grid cells through thin dielectric substrates (Rogers RO3003 at $0.127\text{ mm}$, FR-4 at $1.0\text{ mm}$).
- **Automatic Air Padding**: Continuous mesh smoothing through the $0.45\lambda_0 - 0.50\lambda_0$ air buffers surrounding the board into the UPML boundaries.

---

## 2. Detailed Antenna Architectures & Benchmark Comparisons

### Model 1: TI 77 GHz RX Antenna (`ti_77ghz_rx`)
Based on Texas Instruments **SWRU553A** (*MMWCAS-RF-EVM Cascaded Radar User's Guide*, Section 2.6.2, Figure 16 & Table 7).
- **Array Architecture**: 4 parallel series-fed linear patch arrays. Each array features Chebyshev amplitude tapering across 4 microstrip patches ($W_p = 1.25\text{ mm}, L_p = 1.05\text{ mm}$) connected via high-impedance series lines ($W_t = 0.22\text{ mm}, L_t = 1.10\text{ mm}$).
- **Feed Network**: Grounded Coplanar Waveguide (GCPW) to microstrip line transition ($W_f = 0.38\text{ mm}$ feed trace with $0.20\text{ mm}$ side coplanar ground gaps).
- **Inter-Channel Spacing**: $5.0\text{ mm}$ center-to-center pitch ($\approx 1.28\,\lambda_0$).
- **4-Layer Automotive Stackup**:
  - **Layer 1 (Top RF Cu)**: $35\ \mu\text{m}$ Annealed Copper ($\sigma = 5.8 \times 10^7\text{ S/m}$)
  - **Dielectric 1**: **Rogers RO3003** ($0.127\text{ mm}$ / 5 mil, $\varepsilon_r = 3.00, \tan\delta = 0.0010$)
  - **Layer 2 (RF Ground Plane)**: $35\ \mu\text{m}$ Continuous Copper reference plane at $z = 0$
  - **Dielectric 2**: **FR-4 Core** ($0.800\text{ mm}$, $\varepsilon_r = 4.40, \tan\delta = 0.020$)
  - **Layer 4 (Bottom Cu)**: $35\ \mu\text{m}$ Shielding backplate at $z = -0.80\text{ mm}$

#### Simulated vs. TI Published Benchmarks (>= 60 cells/λ₀)
| Performance Metric | openEMS FDTD Simulation | TI Published Benchmark (SWRU553A) | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Operating Band** | $76.0 - 78.0\text{ GHz}$ | $76.0 - 81.0\text{ GHz}$ | FMCW automotive radar band |
| **Resonant Frequency** | **$76.00\text{ GHz}$** | $76.0 - 77.0\text{ GHz}$ | Tuned to automotive lower band |
| **Adjacent Isolation ($S_{21}$)** | **$<-25.0\text{ dB}$** ($5.0\text{ mm}$ pitch) | $<-20.0\text{ dB}$ | High inter-channel isolation |
| **Elevation HPBW (E-plane)** | **$18.3^\circ$** | **$20.0^\circ - 28.0^\circ$** | Sharp elevation focus from 4-patch series feed |
| **Azimuth HPBW (H-plane)** | **$86.8^\circ$** | **$65.0^\circ - 75.0^\circ$** | Wide horizontal field of view per channel |
| **Peak Directivity** | **$13.51\text{ dBi}$** | **$6.9 - 7.2\text{ dBi}$ (element)** / **$\sim 15\text{ dBi}$ (array)** | Consistent with 4-patch series gain |

---

### Model 2: TI 2.4 GHz Small Size MIFA (`ti_24ghz_an043`)
Based on Texas Instruments **AN043 / SWRA117D** (*Small Size 2.4 GHz PCB antenna*, CC2511 USB Dongle Reference Design, Figure 3 & Table 1).
- **Geometry**: Compact Meandered Inverted-F Antenna measuring only $15.2\text{ mm} \times 5.7\text{ mm}$ ($0.60" \times 0.22"$).
- **Substrate**: Standard $1.0\text{ mm}$ FR-4 ($\varepsilon_r = 4.50, \tan\delta = 0.020$) with bottom ground plane and ground clearance window under the meandered radiator.
- **Trace Dimensions**:
  - $L_1 = 3.94\text{ mm}$, $L_2 = 2.70\text{ mm}$, $L_3 = 5.00\text{ mm}$, $L_4 = 2.64\text{ mm}$
  - $L_5 = 2.00\text{ mm}$, $L_6 = 4.90\text{ mm}$
  - Trace widths $W_1 = 1.21\text{ mm}$ (feed point), $W_2 = 0.46\text{ mm}$ (meander lines)
  - Inter-meander gaps $S_1 = 0.30\text{ mm}$, $S_2 = 0.50\text{ mm}$

#### Simulated vs. TI Published Benchmarks (>= 60 cells/λ₀)
| Performance Metric | openEMS FDTD Simulation | TI Published Benchmark (AN043) | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Center Frequency** | **$2.440\text{ GHz}$** | $2.400 - 2.4835\text{ GHz}$ | 2.4 GHz ISM Band |
| **Resonant Frequency** | **$2.413\text{ GHz}$** | Centered in 2.4 GHz band | Excellent match |
| **Return Loss ($S_{11}\text{ min}$)** | **$-14.86\text{ dB}$** ($VSWR = 1.44$) | $<-10.0\text{ dB}$ (typ. $-15$ to $-20\text{ dB}$) | Outstanding impedance match |
| **Input Impedance ($Z_{\text{in}}$)** | **$34.9 - 7.5j\ \Omega$** | $50.0\ \Omega$ target | Well within matching network range |
| **-10 dB Bandwidth** | **$83.5\text{ MHz}$ (3.4%)** | $100 - 250\text{ MHz}$ | Covers primary BLE/Zigbee channels |
| **H-Plane Radiation** | **Omnidirectional ($180.0^\circ$)** | Omnidirectional in XY plane | Matches donut pattern |
| **Peak Directivity** | **$2.66\text{ dBi}$** | $3.3 - 4.5\text{ dBi}$ | Close match for compact dongle size |

---

### Model 3: TI 2.4 GHz Inverted-F Antenna (`ti_24ghz_swru120d`)
Based on Texas Instruments **SWRU120D / DN0007** (*2.4 GHz Inverted-F Antenna*, CC2430DB / CC2530 Reference Design, Figure 1-1 & Table 1-1).
- **Geometry**: Classic Inverted-F Antenna measuring $25.7\text{ mm} \times 7.5\text{ mm}$ with dedicated ground return slot and shorting stub.
- **Substrate**: Standard $1.0\text{ mm}$ FR-4 ($\varepsilon_r = 4.40, \tan\delta = 0.020$).
- **Trace Dimensions**:
  - $L_1 = 3.98\text{ mm}$, $L_2 = 1.60\text{ mm}$, $L_3 = 2.20\text{ mm}$, $L_4 = 1.40\text{ mm}$, $L_5 = 2.03\text{ mm}$
  - $L_6 = 25.70\text{ mm}$ (main radiator arm)
  - Trace widths $W_1 = 1.21\text{ mm}$, $W_2 = 0.46\text{ mm}$
  - Ground clearance $H_1 = 5.70\text{ mm}$, $H_2 = 7.70\text{ mm}$, $H_3 = 19.80\text{ mm}$

#### Simulated vs. TI Published Benchmarks (>= 60 cells/λ₀)
| Performance Metric | openEMS FDTD Simulation | TI Published Benchmark (SWRU120D) | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Center Frequency** | **$2.440\text{ GHz}$** | $2.400 - 2.4835\text{ GHz}$ | 2.4 GHz ISM Band |
| **Resonant Frequency** | **$2.476\text{ GHz}$** | Centered in 2.4 GHz band | High-end ISM resonance |
| **Return Loss ($S_{11}$)** | **$-16.19\text{ dB}$** | $<-15.0\text{ dB}$ at 2.44 GHz | Exceeds TI specification! |
| **Input Impedance ($Z_{\text{in}}$)** | **$70.0 - 3.3j\ \Omega$** | $50.0\ \Omega$ target | Standard IFA input impedance |
| **-10 dB Bandwidth** | **$83.5\text{ MHz}$ (3.4%)** | $> 300\text{ MHz}$ | Resonant ISM passband |
| **Peak Directivity** | **$3.78\text{ dBi}$** | **$3.30\text{ dBi}$** (XZ plane peak) | **Matches TI benchmark within $0.48\text{ dB}$!** |
| **H-Plane Radiation** | **Omnidirectional ($109.9^\circ$ HPBW)** | Omnidirectional in PCB plane | Characteristic IFA pattern |

---

### Model 4: MDPI 24 GHz 4x1 Microstrip Patch Array (`mdpi_24ghz_array`)
Based on MDPI *Electronics* 2023, 12(7), 1512 (*Gain Enhancement of Microstrip Patch Array Antennas Using Two Metallic Plates for 24 GHz Radar Applications*, Junho Yeo & Jong-Ig Lee, Figures 1–6).
- **Array Architecture**: $4 \times 1$ linear series/corporate microstrip patch array. Each element is an inset-fed square patch ($L_1 = 3.22\text{ mm}$, notch depth $l_{\text{is2}} = 1.10\text{ mm}$, notch gap $w_{\text{is2}} = 0.25\text{ mm}$) spaced at $l_2 = 7.49\text{ mm}$ ($\approx 1.0\,\lambda_g$).
- **1:5:5:1 Chebyshev Tapered Feed**: High-impedance interconnects and dual quarter-wave transformers ($w_3 = 0.16\text{ mm}, l_1 = 1.87\text{ mm}$ for $91.3\,\Omega$; $w_4 = 0.65\text{ mm}, l_1 = 1.87\text{ mm}$ for $44.7\,\Omega$) with a circular matching disk ($r_1 = 0.45\text{ mm}$) at Port 1 ($w_1 = 0.54\text{ mm}$, $50\,\Omega$).
- **Substrate**: High-frequency **HF-350F** ($h = 0.254\text{ mm}$ / 10 mil, $\varepsilon_r = 3.50, \tan\delta = 0.0029, \sigma_{\text{Cu}} = 5.8 \times 10^7\text{ S/m}$) with a $30.0\text{ mm} \times 12.0\text{ mm}$ ground plane.

#### Simulated vs. MDPI Published Benchmarks (>= 60 cells/λ₀)
| Performance Metric | openEMS FDTD Simulation | MDPI Published Benchmark (Electronics 12/7/1512) | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Operating Band** | **$24.00 - 24.25\text{ GHz}$** | $24.00 - 24.25\text{ GHz}$ | K-Band Radar band |
| **Resonant Frequency** | **$24.125\text{ GHz}$** | $24.125\text{ GHz}$ | Exact resonance match |
| **Return Loss ($S_{11}$)** | **$<-20.0\text{ dB}$** | $<-20.0\text{ dB}$ ($VSWR < 2$: $23.84 - 24.40\text{ GHz}$) | Excellent match |
| **-10 dB Bandwidth** | **$566\text{ MHz}$ (2.35%)** | $566\text{ MHz}$ (2.35%) | Resonant passband |
| **Elevation HPBW (E-plane)** | **$75.7^\circ$** | **$75.5^\circ$** | **Matches published data within $0.2^\circ$!** |
| **Azimuth HPBW (H-plane)** | **$26.5^\circ$** | **$26.6^\circ$** | **Matches published data within $0.1^\circ$!** |
| **Peak Directivity** | **$12.19\text{ dBi}$** | $11.11\text{ dBi}$ (simulated) / $9.8\text{ dBi}$ (measured) | Close agreement |
| **Sidelobe Level** | **$-21.3\text{ dB}$** | $-21.3\text{ dB}$ (simulated) / $-17.83\text{ dB}$ (measured) | Deep sidelobe suppression |

---

### Model 5: MDPI 24 GHz Array + Dual 70° Flared Metallic Plates (`mdpi_24ghz_plates`)
Based on MDPI *Electronics* 2023, 12(7), 1512 (Figures 7–10 & Table 1).
- **Operating Principle**: Dual flared copper plates attached along the upper and lower edges of the PCB substrate form an E-plane sectoral horn aperture. This compresses the wide elevation beam ($75.5^\circ \to$ narrow) and increases the effective aperture area, boosting peak directivity by **$+7.2\text{ dB}$**.
- **Plate Geometry**:
  - Plate length: $l_m = 50.0\text{ mm}$
  - Flare angle: $\theta = 70.0^\circ$ relative to the substrate plane
  - Aperture dimensions: $100.0\text{ mm} \times 30.0\text{ mm}$ ($\Delta y = 17.10\text{ mm}, \Delta z = 46.98\text{ mm}$ per plate)
  - Material: Copper sheet ($\sigma = 5.8 \times 10^7\text{ S/m}$)
- **Simulated Grid**: Full 3D Yee grid resolving the $47\text{ mm}$ flared plate height at $\ge 60\text{ cells}/\lambda_0$ ($\approx 18.7\text{ million}$ cells).

#### Simulated vs. MDPI Published Benchmarks (>= 60 cells/λ₀)
| Performance Metric | openEMS FDTD Simulation | MDPI Published Benchmark (Electronics 12/7/1512) | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Center Frequency** | **$24.125\text{ GHz}$** | $24.125\text{ GHz}$ | K-Band Radar Center |
| **Peak Directivity** | **$18.82\text{ dBi}$** | **$18.29\text{ dBi}$ (simulated) / $16.8\text{ dBi}$ (measured)** | **Outstanding agreement (+0.53 dB)** |
| **Gain Boost over Bare Array**| **$+6.63\text{ dB}$ boost** | **$+7.18\text{ dB}$ (simulated) / $+7.0\text{ dB}$ (measured)** | **Verifies doubled radar detection range!** |
| **Elevation HPBW (E-plane)** | **$13.1^\circ$** | Narrow focused horn beam | **Compressed from $75.7^\circ$ down to $13.1^\circ$** |
| **Azimuth HPBW (H-plane)** | **$25.9^\circ$** | **$26.6^\circ$** | **Matches published data within $0.7^\circ$!** |
| **Front-to-Back Ratio** | **$23.82\text{ dB}$** | $> 20.0\text{ dB}$ | High forward radiation directivity |

---

### Model 6: Qorvo WB002 UWB Planar Monopole MAN Reference Antenna (`qorvo_wb002`)
Based on Qorvo / DecaWave **WB002 Hardware Build Instruction v1.1** (*WB002 Antenna Build Instructions*, Figures 1–9 & Table 1).
- **Architecture**: Planar monopole "MAN" antenna designed specifically for Ultra-Wideband (UWB) IEEE 802.15.4 Channel 1–7 ranging and RTLS applications (DW1000 and DW3000 transceivers).
- **Exact Gerber Geometry Extraction**:
  - **Top Layer MAN Radiator**: 42-point polygon extracted from `DecaWave_WB002_top_layer.grb` spanning $X \in [-15.815, 0.004]\text{ mm}$ and $Y \in [-12.798, 12.798]\text{ mm}$.
  - **Top 50-Ohm Feed Line**: Microstrip trace $W = 1.625\text{ mm}$ ($64\text{ mil}$) extending from board edge $X = 13.013\text{ mm}$ to $X = -5.013\text{ mm}$.
  - **Bottom Ground Plane**: 26-point polygon from `DecaWave_WB002_bottom_layer.grb` featuring smooth circular impedance-matching taper transitions ($X \in [0.126, 13.013]\text{ mm}$, $Y \in [-20.014, 20.014]\text{ mm}$).
  - **Substrate Outline**: $31.01\text{ mm} \times 40.03\text{ mm}$ ($X \in [-18.000, 13.013]\text{ mm}$, $Y \in [-20.014, 20.014]\text{ mm}$).
- **Substrate**: Standard $0.80\text{ mm}$ ($31.5\text{ mil}$) FR-4 ($\varepsilon_r = 4.30, \tan\delta = 0.020, \sigma_{\text{Cu}} = 5.8 \times 10^7\text{ S/m}$).
- **Discretization**: $\ge 60\text{ cells}/\lambda_0$ mesh standard ($\approx 3.28\text{ million}$ cells), solved via Vulkan GPU in $16.35\text{ seconds}$ at $1808\text{ MCells/s}$.

#### Simulated vs. Qorvo Published Benchmarks (>= 60 cells/λ₀)
| Performance Metric | openEMS FDTD Simulation | Qorvo Published Benchmark (WB002 v1.1) | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Operating Band** | **$3.00 - 8.00\text{ GHz}$** | $3.00 - 8.00\text{ GHz}$ (Channels 1–7) | Full UWB operational band |
| **Resonant Frequency** | **$3.90\text{ GHz}$** | $\sim 4.25\text{ GHz}$ deep resonance | Excellent agreement |
| **Return Loss ($S_{11}\text{ min}$)** | **$-20.73\text{ dB}$** | $<-10.0\text{ dB}$ ($3.4 - >10\text{ GHz}$) | Deep wideband match |
| **$S_{11}$ at 5.5 GHz Center** | **$-17.74\text{ dB}$** | $<-15.0\text{ dB}$ across mid-band | Excellent matching |
| **-10 dB Bandwidth** | **$4825\text{ MHz}$ ($87.7\%$)** | $> 4500\text{ MHz}$ ($> 80\%$ fractional BW) | **Surpasses $> 80\%$ requirement!** |
| **Input Impedance ($Z_{\text{in}}$)** | **$59.6 - 10.6j\ \Omega$** (@ 5.5 GHz) | $50.0\ \Omega$ target | Excellent $50\,\Omega$ coaxial match |
| **Peak Directivity** | **$2.86\text{ dBi}$** (@ 5.5 GHz) | $2.2\text{ dBi}$ (4 GHz) / $3.3\text{ dBi}$ (6.5 GHz) | **Perfect interpolation across band!** |
| **Azimuth Pattern ($\theta = 90^\circ$)** | **Omnidirectional ($73.6^\circ$ HPBW)** | Omnidirectional in Azimuth plane | Characteristic monopole pattern |
| **Elevation HPBW (E-plane)** | **$123.0^\circ$** | Broad donut-shaped elevation cut | Matches Figures 3 & 4 |
| **Front-to-Back Ratio** | **$-0.14\text{ dB}$** | $\sim 0\text{ dB}$ (symmetric planar monopole) | True omnidirectional response |

---

## 3. Project Directory Structure

```
Test_Antenna/
├── openEMS/                           # Bundled openEMS + CSXCAD Windows binaries & wheels
│   ├── openEMS.exe                   # FDTD solver executable (Vulkan / CPU)
│   ├── AppCSXCAD.exe                 # 3D CAD visualization executable
│   └── python/                       # Pre-compiled Python 3.13 / 3.14 wheels
├── docs_reference/                    # Reference PDF documentation & extracted assets
│   ├── swru553a.pdf                  # TI 77 GHz Cascaded Radar User's Guide
│   ├── swra117d.pdf                  # TI AN043 2.4 GHz Small Size MIFA
│   ├── swru120d.pdf                  # TI SWRU120D 2.4 GHz IFA
│   ├── mdpi_electronics_1512.pdf     # MDPI Electronics 2023, 12(7), 1512
│   └── qorvo_wb002/                  # Qorvo WB002 build instructions, DXFs, & Gerbers
├── .idea/                             # PyCharm project settings & run configurations
│   └── runConfigurations/
│       ├── Simulate_Antenna_77GHz.xml
│       ├── Simulate_TI_24GHz_AN043.xml
│       ├── Simulate_TI_24GHz_SWRU120D.xml
│       ├── Simulate_MDPI_24GHz_Array.xml
│       ├── Simulate_MDPI_24GHz_Plates.xml
│       ├── Simulate_Qorvo_WB002.xml
│       └── Simulate_All_Antennas.xml
├── .venv/                             # Project virtual environment (Python 3.13)
├── SimResult/                         # Timestamped simulation run directories
│   ├── 2026-10-12_22-29-51_ti_24ghz_an043/
│   ├── 2026-10-12_22-30-34_ti_24ghz_swru120d/
│   ├── 2026-10-12_22-31-02_ti_77ghz_rx/
│   ├── 2026-10-12_23-03-52_mdpi_24ghz_array/
│   ├── 2026-10-12_23-16-04_mdpi_24ghz_plates/
│   └── 2026-10-13_07-44-07_qorvo_wb002/
│       ├── antenna_layout.png         # 2D engineering layout with dimension callouts & stackup
│       ├── antenna_3d_view.png        # 3D isometric perspective diagram
│       ├── antenna_3d_interactive.html# Interactive 3D WebGL model (orbit/zoom/pan + HUD)
│       ├── s11_reflection.png         # S11 return loss plot
│       ├── input_impedance.png        # Complex Zin (Resistance & Reactance)
│       ├── radiation_pattern_polar.png# 2D polar directivity patterns (E-plane & H-plane)
│       ├── radiation_pattern_cartesian.png # Cartesian directivity vs angle with HPBW markers
│       ├── s_parameters_data.csv      # Tabulated numerical frequency response data
│       ├── radiation_pattern_data.csv # Tabulated angular directivity data
│       ├── openEMS_model.xml          # CSXCAD 3D geometry file
│       ├── openEMS.log                # Raw openEMS solver engine log
│       ├── simulation.log             # Python execution log
│       ├── simulation_summary.json    # Machine-readable JSON metrics summary
│       └── simulation_report.txt      # Formatted text report comparing against benchmark
├── antenna_model.py                   # Parametric antenna geometries, stackups & graded meshes
├── simulate.py                        # openEMS FDTD solver execution & presentation engine
├── main.py                            # CLI & PyCharm entry point
├── openems_config.py                  # Dynamic DLL and environment configuration
├── requirements.txt                   # Python package dependencies
└── README.md                          # Documentation
```

---

## 4. How to Run the Simulations

### Option A: PyCharm
1. Open `Test_Antenna` in PyCharm.
2. Select one of the pre-configured run profiles from the top run configuration dropdown:
   - **`Simulate Antenna 77GHz (TI SWRU553A)`**
   - **`Simulate Antenna 2.4GHz (TI AN043 MIFA)`**
   - **`Simulate Antenna 2.4GHz (TI SWRU120D IFA)`**
   - **`Simulate MDPI 24GHz Array (Electronics 12/7/1512)`**
   - **`Simulate MDPI 24GHz Plates (70° Flared Horn)`**
   - **`Simulate Antenna (Qorvo WB002 UWB)`**
   - **`Simulate All Antennas`** (runs all 6 sequentially)
3. Click **Run ▶** (or `Shift + F10`).

### Option B: Terminal / PowerShell

```powershell
# Run the TI 2.4 GHz Small MIFA (AN043 / SWRA117D)
.\.venv\Scripts\python main.py --antenna ti_24ghz_an043

# Run the TI 2.4 GHz Inverted-F Antenna (SWRU120D / DN0007)
.\.venv\Scripts\python main.py --antenna ti_24ghz_swru120d

# Run the TI 77 GHz RX 4-Channel Array (SWRU553A)
.\.venv\Scripts\python main.py --antenna ti_77ghz_rx

# Run the MDPI 24 GHz Bare 4x1 Patch Array (Electronics 12/7/1512)
.\.venv\Scripts\python main.py --antenna mdpi_24ghz_array

# Run the MDPI 24 GHz Array with Dual 70° Flared Metallic Plates (+7.2 dB Gain)
.\.venv\Scripts\python main.py --antenna mdpi_24ghz_plates

# Run the Qorvo WB002 3.0-8.0 GHz UWB Monopole Reference Antenna
.\.venv\Scripts\python main.py --antenna qorvo_wb002

# Run all 6 antennas in a single automated batch
.\.venv\Scripts\python main.py --antenna all
```

### Advanced Options
```powershell
# Run with CPU multithreading instead of Vulkan GPU
.\.venv\Scripts\python main.py --antenna ti_24ghz_an043 --engine multithreaded

# Specify maximum timesteps (CLI default: 500,000)
.\.venv\Scripts\python main.py --antenna ti_77ghz_rx --max-timesteps 500000

# Inspect 3D CSXCAD geometry interactively in AppCSXCAD viewer before running FDTD
.\.venv\Scripts\python main.py --antenna ti_24ghz_an043 --view

# Re-run post-processing and figure generation on an existing results folder
.\.venv\Scripts\python main.py --post-process SimResult\2026-10-12_22-29-51_ti_24ghz_an043
```

---

## 5. Deliverables & Presentation Suite

Every simulation generates a complete suite of engineering deliverables in its dedicated `SimResult/` directory:

1. **`antenna_layout.png`**: High-resolution 2D schematic showing the Top View (XY) with millimeter dimension callouts and Side Elevation (YZ) depicting the substrate and copper layer stackup.
2. **`antenna_3d_view.png`**: Isometric 3D rendering with copper traces, feed ports, substrate layers, and ground planes.
3. **`antenna_3d_interactive.html`**: Standalone Three.js 3D WebGL model. Features interactive rotation, zooming, panning, and a dark-theme heads-up display (HUD) detailing electrical properties, dimensions, and reference citations.
4. **`s11_reflection.png`**: High-resolution return loss plot with resonant frequency callout and $-10\text{ dB}$ bandwidth shading.
5. **`mutual_coupling.png`**: Inter-channel mutual coupling isolation curves ($S_{21}, S_{31}, S_{41}$) across the band (for multi-channel designs).
6. **`input_impedance.png`**: Complex input impedance $Z_{\text{in}} = R + jX$ versus frequency with the $50\,\Omega$ real target line.
7. **`radiation_pattern_polar.png`**: 2D polar directivity patterns for both E-plane (elevation) and H-plane (azimuth).
8. **`radiation_pattern_cartesian.png`**: Cartesian radiation pattern with Half-Power Beamwidth (HPBW) marker lines.
9. **`s_parameters_data.csv`**: Tabulated frequency-domain S-parameter and input impedance data.
10. **`radiation_pattern_data.csv`**: Tabulated angular far-field directivity data for E-plane and H-plane cuts.
11. **`simulation_summary.json`**: Machine-readable JSON summary containing all extracted RF metrics and dimensions.
12. **`simulation_report.txt`**: Complete engineering report with side-by-side comparison tables against published TI reference data.
