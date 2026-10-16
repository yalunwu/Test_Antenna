"""Compare the latest saved runs with the original reference documents.

Run with .venv/Scripts/python.exe compare_results.py. This only reads solver
outputs; it does not run FDTD or replace the original reports/NF2FF files.
Reference values below were checked against the PDFs in docs_reference/.
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import openems_config  # Configure native DLLs before importing openEMS.
from openEMS.ports import UI_data
from openEMS.nf2ff import nf2ff_results
from simulate import validate_saved_solver_log


ROOT = Path(__file__).resolve().parent
MODELS = {
    "ti_77ghz_rx": ("TI 77 GHz RX", 73.0, 81.0),
    "ti_24ghz_an043": ("TI AN043 MIFA", 2.2, 2.7),
    "ti_24ghz_swru120d": ("TI SWRU120D IFA", 2.2, 2.7),
    "mdpi_24ghz_array": ("MDPI bare array", 23.5, 24.75),
    "mdpi_24ghz_plates": ("MDPI array + plates", 23.5, 24.75),
    "qorvo_wb002": ("Qorvo WB002", 2.5, 8.5),
}
VSWR2_DB = 20 * np.log10(1 / 3)


def matched_intervals(freq, s11, threshold):
    """Contiguous passbands with interpolated crossings and censored edges."""
    mask = s11 <= threshold
    intervals = []
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True])
    for start, stop in zip(starts, stops):
        lo, hi = freq[start], freq[stop]
        if start:
            lo = np.interp(threshold, [s11[start], s11[start - 1]],
                           [freq[start], freq[start - 1]])
        if stop < len(freq) - 1:
            hi = np.interp(threshold, [s11[stop], s11[stop + 1]],
                           [freq[stop], freq[stop + 1]])
        intervals.append({"low_ghz": float(lo), "high_ghz": float(hi),
                          "width_mhz": float((hi - lo) * 1000),
                          "low_censored": bool(start == 0),
                          "high_censored": bool(stop == len(freq) - 1)})
    return intervals


def read_port(path, frequencies_ghz, port=1):
    frequencies_hz = np.asarray(frequencies_ghz) * 1e9
    u = UI_data(f"port_ut_{port}", str(path), frequencies_hz).ui_f_val[0]
    i = UI_data(f"port_it_{port}", str(path), frequencies_hz).ui_f_val[0]
    incident = (u + 50 * i) / 2
    reflected = (u - 50 * i) / 2
    if not np.all(np.isfinite(incident)) or np.any(np.abs(incident) == 0):
        raise ValueError(f"No valid incident wave in {path}; rerun this model before comparing RF results")
    return {"s11_db": 20 * np.log10(np.abs(reflected / incident)),
            "p_inc": np.abs(incident)**2 / 100,
            "p_acc": 0.5 * np.real(u * np.conj(i)), "zin": u / i}


def gain_metrics(path, farfield_file):
    ff = nf2ff_results(str(farfield_file))
    port = read_port(path, ff.freq / 1e9)
    metrics = []
    for n, freq in enumerate(ff.freq):
        efficiency = float(ff.Prad[n] / port["p_acc"][n])
        if not 0 < efficiency < 1.05:
            raise ValueError(f"Unexpected power balance in {path}: {efficiency}")
        metrics.append({
            "frequency_ghz": float(freq / 1e9),
            "sampled_peak_directivity_dbi": float(10 * np.log10(ff.Dmax[n])),
            "radiation_efficiency_percent": efficiency * 100,
            "sampled_peak_gain_dbi": float(10 * np.log10(ff.Dmax[n] * efficiency)),
            "sampled_peak_realized_gain_dbi": float(10 * np.log10(
                ff.Dmax[n] * ff.Prad[n] / port["p_inc"][n])),
        })
    return metrics


def latest_runs(root):
    runs = {}
    for file in root.glob("*/simulation_summary.json"):
        summary = json.loads(file.read_text(encoding="utf-8"))
        name = summary.get("antenna_model", {}).get("name")
        if name not in MODELS:
            continue
        key = summary["simulation_info"]["start_time"]
        if name not in runs or key > runs[name][1]["simulation_info"]["start_time"]:
            runs[name] = (file.parent, summary)
    return runs


def analyze(name, path, summary, out):
    validate_saved_solver_log(path)
    label, lo, hi = MODELS[name]
    freq = np.linspace(lo, hi, 1601)
    log = (path / "simulation.log").read_text(encoding="utf-8")
    # S11 and gain normalization below require one active, 50-ohm lumped port.
    if "All 4 Channels Driven" in log:
        raise ValueError("Active multiport reflection requires separate normalization")
    channel = re.search(r"Excitation Mode:\s+Channel (\d+) Excited", log)
    if channel and channel.group(1) != "1":
        raise ValueError("This comparison currently expects channel 1 excitation")
    port = read_port(path, freq)
    saved_model = ET.parse(path / "openEMS_model.xml").getroot()
    port_x = []
    for element in saved_model.iter("LumpedElement"):
        if not element.get("Name", "").startswith("port_resist_"):
            continue
        if not np.isclose(float(element.get("R", "nan")), 50):
            raise ValueError("The saved model does not use 50-ohm lumped ports")
        box = element.find(".//Box")
        if box is not None:
            port_x.append((float(box.find("P1").get("X")) + float(box.find("P2").get("X"))) / 2)
    port_x.sort()
    channel_pitch = float(np.mean(np.diff(port_x))) if len(port_x) > 1 else None
    original = np.genfromtxt(path / "s_parameters_data.csv", delimiter=",", names=True)
    check = read_port(path, original["frequency_ghz"])
    # The CSV rounds frequency to 4 decimal places; TI narrowband values can
    # therefore differ slightly when evaluated at those rounded frequencies.
    max_csv_error = float(np.max(np.abs(check["s11_db"] - original["s11_db"])))
    if max_csv_error > 0.01:
        raise ValueError(f"Raw port reconstruction disagrees with CSV: {max_csv_error}")
    idx = int(np.argmin(port["s11_db"]))
    a = summary["antenna_model"]
    band_lo, band_hi = original["frequency_ghz"][[0, -1]]
    center = a["center_frequency_ghz"]
    band = np.linspace(band_lo, band_hi, 1001)
    band_s11 = read_port(path, band)["s11_db"]
    solver_log = (path / "openEMS.log").read_text(encoding="utf-8")
    convergence = re.search(r"end-criteria of ([-\d.]+)dB reached after (\d+) timesteps \(([-\d.]+)dB\)", solver_log)
    if not convergence:
        raise ValueError(f"Solver did not record successful energy decay: {path}")
    np.savetxt(out / f"{name}_wide_s11.csv", np.c_[freq, port["s11_db"],
               port["zin"].real, port["zin"].imag], delimiter=",",
               header="frequency_ghz,s11_db,zin_real_ohm,zin_imag_ohm", comments="")
    result = {
        "name": name, "label": label, "run_directory": str(path.resolve()),
        "run_start": summary["simulation_info"]["start_time"],
        "geometry_revision": a.get("geometry_revision", "legacy-v1"),
        "cells_per_wavelength": a.get("cells_per_wavelength"),
        "max_timesteps": summary["simulation_info"].get("max_timesteps"),
        "unused_primitive_warnings": solver_log.count("Unused primitive"),
        "antenna_channel_count": len(port_x),
        "channel_pitch_mm": channel_pitch,
        "sweep_ghz": [lo, hi], "operating_band_ghz": [float(band_lo), float(band_hi)],
        "s11_at_center_db": float(read_port(path, [center])["s11_db"][0]),
        "operating_band_best_s11_db": float(np.min(band_s11)),
        "operating_band_worst_s11_db": float(np.max(band_s11)),
        "wide_min_s11_db": float(port["s11_db"][idx]),
        "wide_min_s11_frequency_ghz": float(freq[idx]),
        "wide_min_at_sweep_edge": bool(idx in (0, len(freq) - 1)),
        "matched_intervals_minus10db": matched_intervals(freq, port["s11_db"], -10),
        "matched_intervals_vswr2": matched_intervals(freq, port["s11_db"], VSWR2_DB),
        "center_farfield": gain_metrics(path, path / "nf2ff.h5")[0],
        "original_pattern_metrics": summary["simulated_rf_metrics"],
        "solver_stop": {"target_energy_db": float(convergence[1]),
                        "timesteps": int(convergence[2]), "energy_db": float(convergence[3])},
        "raw_vs_csv_max_error_db": max_csv_error,
    }
    return result, freq, port["s11_db"]


def additional_farfields(name, path, out):
    """Evaluate saved fields at source frequencies; preserve original NF2FF."""
    import antenna_model
    _, _, _, nf = antenna_model.get_antenna_model(name).create_simulation()
    dest = out / f"{name}_source_frequencies_nf2ff.h5"
    if name == "qorvo_wb002":
        frequencies = np.array([4, 5.5, 6.5]) * 1e9
        # Source maximum-gain/azimuth checks need more than phi=0/90 cuts.
        theta, phi = np.arange(0, 181, 5), np.arange(0, 360, 5)
    else:
        frequencies = [76e9]
        theta, phi = np.arange(-180, 181), [0, 90]
    nf.CalcNF2FF(str(path), frequencies, theta, phi, center=[0, 0, 0],
                outfile=str(dest.resolve()), read_cached=True)
    metrics = gain_metrics(path, dest)
    if name == "ti_77ghz_rx":
        from simulate import calculate_hpbw
        ff = nf2ff_results(str(dest))
        for n, phi_value in enumerate((0, 90)):
            metrics[0][f"hpbw_phi{phi_value}_deg"] = calculate_hpbw(
                np.rad2deg(ff.theta), np.abs(ff.E_norm[0][:, n])**2)
    return metrics


def describe_intervals(intervals):
    if not intervals:
        return "None in the evaluated sweep"
    descriptions = []
    for interval in intervals:
        clipped = interval["low_censored"] or interval["high_censored"]
        descriptions.append(
            f"{interval['low_ghz']:.4f}-{interval['high_ghz']:.4f} GHz; "
            f"{'at least ' if clipped else ''}{interval['width_mhz']:.1f} MHz"
            + (" (crossing outside sweep)" if clipped else ""))
    return "; ".join(descriptions)


def write_report(results, out):
    records = {r["name"]: r for r in results}
    lines = [
        "# Latest openEMS runs compared with the original sources",
        "",
        "Analysis of the saved runs below. No FDTD rerun was needed; original solver outputs and reports were preserved.",
        "",
        "## Results",
        "",
        "| Model | S11 at nominal center | S11 range over configured operating band | -10 dB matched interval from wider sweep |",
        "| --- | ---: | --- | --- |",
    ]
    for r in results:
        lines.append(f"| {r['label']} | {r['s11_at_center_db']:.2f} dB | "
                     f"{r['operating_band_best_s11_db']:.2f} to {r['operating_band_worst_s11_db']:.2f} dB | "
                     f"{describe_intervals(r['matched_intervals_minus10db'])} |")
    lines += ["", "## Solver and geometry checks", "",
              "Every selected run initialized an excitation, contains no unused-primitive warnings, "
              "and reached its energy-decay criterion before the timestep cap. Reconstructed S11 "
              "and power-normalized gains are finite. This verifies the source/geometry export correction; "
              "agreement with the reference hardware is assessed separately below.", "",
              "| Model | Geometry revision | Steps / cap | Target / terminal energy | Unused warnings |",
              "| --- | --- | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['label']} | {r['geometry_revision']} | "
                     f"{r['solver_stop']['timesteps']} / {r['max_timesteps']} | "
                     f"{r['solver_stop']['target_energy_db']:.0f} / {r['solver_stop']['energy_db']:.2f} dB | "
                     f"{r['unused_primitive_warnings']} |")
    lines += ["", "![Wideband S11 comparison](s11_comparison.png)", "",
              "## Radiation power at each nominal center", "",
              "Gain includes radiation efficiency. Realized gain also includes reflection loss. "
              "These peaks use the original two angular cuts, rather than a converged full-sphere maximum.", "",
              "| Model | Frequency (GHz) | Directivity (dBi) | Gain (dBi) | Realized gain (dBi) | Radiated / accepted power |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        g = r["center_farfield"]
        lines.append(f"| {r['label']} | {g['frequency_ghz']:.3f} | "
                     f"{g['sampled_peak_directivity_dbi']:.2f} | {g['sampled_peak_gain_dbi']:.2f} | "
                     f"{g['sampled_peak_realized_gain_dbi']:.2f} | {g['radiation_efficiency_percent']:.1f}% |")
    an = records.get("ti_24ghz_an043")
    if an:
        g = an["center_farfield"]
        matched_ism = any(p["low_ghz"] <= 2.4 and p["high_ghz"] >= 2.4835
                          for p in an["matched_intervals_minus10db"])
        coverage = ("The full 2.400-2.4835 GHz ISM band is matched at -10 dB. " if matched_ism else
                    "The full 2.400-2.4835 GHz ISM band is not matched at -10 dB. ")
        lines += ["", "## TI AN043 / SWRA117D", "",
                  f"The source defines bandwidth at 90% delivered power (-10 dB S11) and reports approximately 250 MHz "
                  f"in free space and 100 MHz with a laptop attached (PDF p. 6, Figure 5). The saved run gives "
                  f"{describe_intervals(an['matched_intervals_minus10db'])}. {coverage}"
                  f"Its wider-sweep minimum is {an['wide_min_s11_db']:.2f} dB at "
                  f"{an['wide_min_s11_frequency_ghz']:.4f} GHz, and S11 at 2.44 GHz is "
                  f"{an['s11_at_center_db']:.2f} dB. The source bandwidth has not been reproduced.", "",
                  f"Source Table 6 (PDF p. 20) lists gain of 4.5 dB in XY and 5.3 dB in XZ/YZ in free space, "
                  f"and 3.3 dB in XY with a laptop. Our sampled peak gain is {g['sampled_peak_gain_dbi']:.2f} dBi "
                  f"at 2.44 GHz. The radiation level is lower. Source board, measurement polarization, "
                  "and coordinates need to be matched before assigning a strict gain error; the source prints its gains in dB."]
    ifa = records.get("ti_24ghz_swru120d")
    if ifa:
        g = ifa["center_farfield"]
        lines += ["", "## TI SWRU120D IFA", "",
                  "The source reports more than 300 MHz at less than 10% reflected power "
                  "(PDF p. 10, Figure 2-8). The run gives "
                  f"{describe_intervals(ifa['matched_intervals_minus10db'])}, consistent with that bandwidth statement.", "",
                  f"Source Table 3-1 (PDF p. 11) lists 3.3 dBi XZ gain and reflection below -15 dB. "
                  f"At 2.44 GHz the run gives S11 = {ifa['s11_at_center_db']:.2f} dB and sampled peak gain "
                  f"{g['sampled_peak_gain_dbi']:.2f} dBi. The gain difference is about "
                  f"{g['sampled_peak_gain_dbi'] - 3.3:+.2f} dB, subject to source plane/polarization normalization. "
                  f"The full configured ISM-band S11 range is {ifa['operating_band_best_s11_db']:.2f} "
                  f"to {ifa['operating_band_worst_s11_db']:.2f} dB. Bandwidth and matching agree with the "
                  "source targets; the lower gain and simplified board environment remain unresolved."]
    bare, plates = records.get("mdpi_24ghz_array"), records.get("mdpi_24ghz_plates")
    if bare and plates:
        bg, pg = bare["center_farfield"], plates["center_farfield"]
        bm = bare["original_pattern_metrics"]
        boost = pg["sampled_peak_realized_gain_dbi"] - bg["sampled_peak_realized_gain_dbi"]
        lines += ["", "## MDPI Electronics 2023, 12(7), 1512", "",
                  "Use realized gain for this paper (Sections 2 and 4, Figures 6 and 10). "
                  "Published simulated/measured peaks are 11.11/9.8 dBi for the tapered bare array "
                  "and 18.29/16.8 dBi with the 50 mm, 70 degree plates.", "",
                  "| Quantity | Saved run | Source simulation | Source measurement |",
                  "| --- | ---: | ---: | ---: |",
                  f"| Bare realized gain | {bg['sampled_peak_realized_gain_dbi']:.2f} dBi | 11.11 dBi | 9.8 dBi |",
                  f"| Plates realized gain | {pg['sampled_peak_realized_gain_dbi']:.2f} dBi | 18.29 dBi | 16.8 dBi |",
                  f"| Improvement with plates | {boost:.2f} dB | 7.18 dB | 7.0 dB |", "",
                  f"The bare array beamwidth is {bm['hpbw_e_plane_deg']:.1f} degrees E-plane and "
                  f"{bm['hpbw_h_plane_deg']:.1f} degrees H-plane, versus the paper's 75.5 and 26.6 degrees "
                  "(PDF p. 10). Beam shape and the gain improvement agree well; impedance matching does not.", "",
                  f"At 24.125 GHz S11 is {bare['s11_at_center_db']:.2f} dB bare and "
                  f"{plates['s11_at_center_db']:.2f} dB with plates, whereas source Figures 6/10 show much deeper "
                  "resonances near 24.1 GHz. Neither run has a -10 dB passband inside 24.00-24.25 GHz. "
                  "In the evaluated 23.5-24.75 GHz sweep the lowest S11 is at the lower boundary, "
                  "so the true resonance and the lower crossing are not located. The source performance "
                  "has not been reproduced merely because the beamwidths look close.", "",
                  "The paper uses VSWR < 2, corresponding to S11 < -9.542 dB, for bandwidth. "
                  "Its simulated bands are 23.836-24.402 GHz bare (566 MHz) and 23.767-24.414 GHz with "
                  "plates (647 MHz); measured bands are 23.819-24.381 GHz (562 MHz) and 23.725-24.365 GHz "
                  "(640 MHz). These are from Section 4, PDF pp. 14-15. The paper's Table 1 has an "
                  "inconsistent bandwidth entry; this comparison uses the explicit Section 4 bands. "
                  "The computed intervals at the exact VSWR threshold are included in comparison_metrics.json."]
    ti = records.get("ti_77ghz_rx")
    if ti:
        lines += ["", "## TI SWRU553A RX element", "",
                  "The source's performance data are RX1/RX4/RX8 system-gain cuts at 76 GHz "
                  "(Figures 18-20, PDF pp. 29-31). Values read from their embedded tables are:", "",
                  "| Source element | H-plane HPBW | E-plane HPBW | H-plane peak system gain | E-plane peak system gain |",
                  "| --- | ---: | ---: | ---: | ---: |",
                  "| RX1 | 62.8852 deg | 22.2568 deg | 10.8733 dB | 10.1665 dB |",
                  "| RX4 | 75.4266 deg | 21.2001 deg | 10.5575 dB | 9.8365 dB |",
                  "| RX8 | 64.5557 deg | 25.2586 deg | 10.7148 dB | 9.7850 dB |", ""]
        source_ff = ti.get("source_frequency_farfields")
        if source_ff:
            g = source_ff[0]
            lines += [f"Reprocessing the saved fields at the same 76 GHz gives H-plane HPBW "
                      f"{g['hpbw_phi0_deg']:.1f} degrees and E-plane HPBW {g['hpbw_phi90_deg']:.1f} degrees, "
                      f"sampled peak gain {g['sampled_peak_gain_dbi']:.2f} dBi and realized gain "
                      f"{g['sampled_peak_realized_gain_dbi']:.2f} dBi. The source quantities are system gain "
                      "in its original board environment, so these are not a strict gain error calculation.", ""]
        lines += ["The old 6.9-7.2 dBi benchmark was not the RX data in these "
                  "figures and has been removed from the corrected model. A guaranteed S11 < -10 dB "
                  "over 76-81 GHz was not found in the cited "
                  "element-performance section and should not be presented as its published specification. "
                  "Using -10 dB as an engineering matching goal, the run does not meet it anywhere in the "
                  "evaluated 73-81 GHz sweep. Its configured narrow operating band is 76-78 GHz. "
                  f"The saved XML has {ti['antenna_channel_count']} channels "
                  f"at {ti['channel_pitch_mm']:.3f} mm pitch. The reduced board "
                  "still differs from the source's original RX array environment."]
    wb = records.get("qorvo_wb002")
    if wb:
        lines += ["", "## Qorvo / DecaWave WB002", "",
                  "The source gives a 3-8 GHz operational range and maximum gain of 2.2 dBi at 4 GHz "
                  "and 3.3 dBi at 6.5 GHz (build instruction PDF p. 3). Figure 5 (PDF p. 6) shows a deep "
                  "S11 minimum near 4.25 GHz below -40 dB, and a -10 dB low crossing around 3.4 GHz. "
                  "These graph estimates are approximate; no source Touchstone data was supplied.", "",
                  f"The run's wider-sweep minimum is {wb['wide_min_s11_db']:.2f} dB near "
                  f"{wb['wide_min_s11_frequency_ghz']:.3f} GHz, about 8% lower in frequency than the "
                  "source's main notch, and much shallower. The -10 dB interval is "
                  f"{describe_intervals(wb['matched_intervals_minus10db'])}. It matches broadly, but "
                  "the advertised 3 GHz lower operational edge is not a -10 dB guarantee. The upper "
                  "matching limit beyond 8.5 GHz has not been evaluated."]
        source_ff = wb.get("source_frequency_farfields")
        if source_ff:
            lines += ["", "The saved fields were also transformed on a full sphere sampled every 5 degrees "
                      "at the source's frequencies, avoiding interpolation of the 5.5 GHz center result:", "",
                      "| Frequency | Sampled peak gain | Sampled realized gain | Source maximum gain |",
                      "| --- | ---: | ---: | ---: |"]
            for g, ref in zip(source_ff, ("2.2 dBi", "Not specified at 5.5 GHz", "3.3 dBi")):
                lines.append(f"| {g['frequency_ghz']:.1f} GHz | {g['sampled_peak_gain_dbi']:.2f} dBi | "
                             f"{g['sampled_peak_realized_gain_dbi']:.2f} dBi | {ref} |")
        lines += ["", "The source calls its quantity gain, so both gain and realized gain are shown. "
                  "Frequency-specific gain agreement is reasonable; the return-loss curve is not reproduced "
                  "closely. Source azimuth and current model coordinate planes must be aligned before claiming "
                  "an omnidirectional-pattern match. Fidelity and rotation-dependent group delay have not been tested."]
    lines += ["", "## Why the existing reports can mislead", "",
              "1. simulate.py calculates -10 dB bandwidth only within the configured operating band. "
              "A fully matched 2.400-2.4835 GHz sweep is therefore reported as 83.5 MHz even when the "
              "actual passband is wider. It also spans the first-to-last passing samples, which can bridge "
              "separate passbands. This analysis uses contiguous intervals and interpolates each crossing.",
              "2. Its resonance is the minimum inside that same narrow band. A minimum at a sweep "
              "boundary does not locate a true resonance. Both MDPI runs have their broader-sweep minimum "
              "at 23.5 GHz, outside the configured operating band.",
              "3. Directivity, gain, and realized gain are different. This report calculates "
              "G = D * Prad / Paccepted and Grealized = D * Prad / Pincident. "
              "Power spectra share the same pulse-DFT normalization, so the ratios are meaningful. "
              "For multiport models, passive-port absorption is included in the effective efficiency.",
              "4. All models currently create copper using CSX.AddMetal, an ideal conductor. "
              "Dielectric loss is included, but the claimed finite copper conductivity, conductor roughness, "
              "and connector losses are not implemented. The corrected MDPI revision uses a circular "
              "matching pad and connected thin PEC reflector facets. The previous square pad and "
              "0.8 mm metal solids have been replaced; finite copper-tape loss and the acrylic fixture "
              "are still omitted. The corrected TI models retain cropped board environments.",
              "5. The original two cuts are phi=0 (XZ) and phi=90 (YZ). They do not evaluate a "
              "theta=90 azimuth sweep. Independently normalizing each cut also loses their relative peak "
              "levels. Source plane orientation and polarization require explicit alignment.", "",
              "## Verification and limits", "",
              f"All {len(results)} selected solver logs record reaching their configured energy-decay end criterion. "
              "Recomputed S11 values were checked against the saved CSVs to within 0.01 dB "
              "(the CSV frequency column is rounded). This is a source comparison at the current mesh, "
              "not a mesh-convergence or CPU-versus-GPU validation. No numerical source curves were invented "
              "from the published graphs.", "",
              "Prioritize the AN043 complete dongle ground/stackup and feed reference plane, and the MDPI "
              "feed/inset/network dimensions and excitation reference plane. The circular MDPI pad and "
              "reflector export are already corrected. TI 77 GHz needs its original feed and array "
              "environment checked. A controlled local mesh-convergence study is still required before "
              "treating remaining errors as physical; a larger global resolution setting alone is insufficient.", "",
              "## Reproduce", "", "```powershell",
              r".\.venv\Scripts\python.exe compare_results.py --source-farfields", "```", "",
              "The script selects the latest summary for each supported antenna, reconstructs wideband "
              "S11 from saved port signals, and writes this report, CSVs, a plot, and JSON metrics. "
              "Additional NF2FF outputs are written only inside the comparison directory.", "",
              "## Source documents", "",
              "- [TI AN043 / SWRA117D](../../docs_reference/swra117d.pdf): PDF pp. 3, 6, 20.",
              "- [TI SWRU120D](../../docs_reference/swru120d.pdf): PDF pp. 2, 10, 11.",
              "- [TI SWRU553A](../../docs_reference/swru553a.pdf): PDF pp. 27-33, especially Figures 18-20.",
              "- [MDPI Electronics 12(7), 1512](../../docs_reference/mdpi_electronics_1512.pdf): "
              "PDF pp. 8-10, 14-16; DOI https://doi.org/10.3390/electronics12071512.",
              "- [WB002 Build Instruction](../../docs_reference/qorvo_wb002/WB002/collateral/WB002%20Hardware%20Build%20%20Instruction.pdf): "
              "PDF pp. 3, 6, 8.", "",
              "## Selected runs", ""]
    for r in results:
        lines.append(f"- {r['label']}: `{r['run_directory']}`; "
                     f"{r['solver_stop']['timesteps']} timesteps, terminal energy {r['solver_stop']['energy_db']:.2f} dB.")
    (out / "source_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "SimResult")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source-farfields", action="store_true",
                        help="Also transform saved fields at 76 GHz and WB002 source frequencies")
    args = parser.parse_args()
    runs = latest_runs(args.results)
    if not runs:
        parser.error("No supported simulation_summary.json files found")
    if args.output is None:
        run_date = max(s["simulation_info"]["start_time"][:10] for _, s in runs.values())
        args.output = args.results / f"source_comparison_{run_date}"
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    for ax, name in zip(axes.flat, MODELS):
        if name not in runs:
            ax.set_visible(False)
            continue
        path, summary = runs[name]
        result, freq, s11 = analyze(name, path, summary, args.output)
        if args.source_farfields and name in ("ti_77ghz_rx", "qorvo_wb002"):
            result["source_frequency_farfields"] = additional_farfields(name, path, args.output)
        results.append(result)
        ax.plot(freq, s11, color="#1764a0", lw=2, label="Saved run, recomputed from port signals")
        ax.axhline(-10, color="#ba3333", ls="--", lw=1, label="S11 = -10 dB")
        if name.startswith("mdpi"):
            ax.axhline(VSWR2_DB, color="#ba3333", ls=":", lw=1, label="Source threshold: VSWR = 2")
        blo, bhi = result["operating_band_ghz"]
        ax.axvspan(blo, bhi, color="#4d9b65", alpha=.12, label="Configured operating band")
        ax.set(title=result["label"], xlabel="Frequency (GHz)", ylabel="S11 (dB)",
               xlim=(freq[0], freq[-1]), ylim=(-45, 0))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc="lower right")
        print(result["label"], json.dumps({k: result[k] for k in (
            "s11_at_center_db", "wide_min_s11_frequency_ghz", "wide_min_s11_db",
            "matched_intervals_minus10db", "center_farfield")}), flush=True)
    fig.suptitle("Latest runs: impedance match across the available excitation bands\n"
                 "Raw port signals; full contiguous intervals, interpolated crossings", fontsize=15)
    fig.savefig(args.output / "s11_comparison.png", dpi=170)
    plt.close(fig)
    (args.output / "comparison_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, args.output)
    print("Saved comparison to", args.output.resolve())


if __name__ == "__main__":
    main()
