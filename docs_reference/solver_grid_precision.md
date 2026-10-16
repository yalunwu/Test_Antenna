# Solver XML coordinate precision correction

The October 13 AN043 run at `09-28-13` recorded zero voltage and current for its entire duration. Its unused `port_excite_1` warning identifies a missing excitation, rather than a slowly decaying antenna. Raising the resolution or timestep cap cannot repair that run.

The bundled CSXCAD exporter writes primitive coordinates with six decimal places in scientific notation and writes mesh coordinates with more digits. The AN043 source's X position became `-4.535870e+00`, while its mesh line was `-4.53586999999999`. Those values differ by only about 1e-14 mm, but a zero-width source must coincide exactly with its grid line. The same serialization mismatch left 616 metal primitives unused in the `09-34-53` MDPI plates run. Both runs are invalid for RF comparison; their original files are preserved.

`solver_xml.write_solver_xml` restores Box/Cylinder endpoints and mesh lines from the original double values using 17 significant digits. Antenna dimensions and mesh density remain the same. Both the normal simulation pipeline and geometry validator use this exporter. The validator now reloads the saved XML before constructing native operators and checks excitation as well as metal usage; the previous validation constructed the model in memory and missed serialization failures.

The solver runner stops when a source or metal primitive is unused. Post-processing and source comparisons reject such saved logs and missing incident waves; post-processing also rejects zero or non-finite driven-port recordings instead of presenting NaN RF metrics as a successful result. An executable failure is reported directly rather than silently retrying a different, in-memory geometry path.

Verification commands:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\validate_geometry.py --native --output SimResult\geometry_validation_xml
.\.venv\Scripts\python.exe main.py --antenna ti_24ghz_an043 --max-timesteps 500000
```

The 13 regression checks include exact grid/Box coordinate preservation across XML reloading for all six models, thin-sheet placement, rejection of zero recordings, and termination on an unused source. Native validation results are saved under `SimResult/geometry_validation_xml`. A larger timestep cap allows the corrected, excited model to reach its existing -40 dB energy stopping criterion; it does not change mesh resolution or establish mesh convergence.

Verified rerun: `SimResult/2026-10-13_09-47-40_ti_24ghz_an043`, 60 cells per free-space wavelength, 1,179,488 grid cells. The standalone Vulkan solver initialized four excitation voltage points, emitted zero unused-primitive warnings, and reached -43.57 dB energy at 143,450 steps (84.72 seconds of time stepping). Both port recordings are nonzero and all S11/impedance samples are finite. The cap was 500,000 steps; the solver stopped on energy decay.

RF agreement is still unresolved: S11 at 2.44 GHz is -4.13 dB, and the minimum over 2.2–2.7 GHz is approximately -7.40 dB at 2.2816 GHz. This corrects the solver-input failure; it does not validate agreement with TI's measured antenna. The MDPI plates model passed the corrected native XML setup, but its previously invalid RF run still needs a fresh simulation before comparison.
