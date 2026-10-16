AI disclosure: Drafted with Codex from the current openEMS and CSXCAD sources.

# Complete HIP GPU setup

Status: proposed; implementation milestones below are pending.
Date: 2026-10-16.
Baseline: openEMS `2a0e069fcdcaed04373ae9f29a82846b0711c149`, CSXCAD
`c46e3206f20027b0b4605c797d07bddab76de942`, openEMS-Project
`38535b35e9afd389f6fbfce3a275285d2f080626`.

## Outcome and scope

Support every built-in numerical setup operation on HIP, for valid models within
the GPU engine's indexing and memory limits. Keep the resulting operator,
extension data and engine initialization data on the device through the first
timestep. Preserve CPU construction as the reference and fallback.

This includes geometry discretization, material lookup and averaging, timestep
selection, boundaries, excitation, extensions, cylindrical/multigrid setup,
coefficient packing and numerical sampling-table construction. Python/Matlab
model generation, XML/STL/PLY/HDF5 decoding, expression parsing, file output and
launch orchestration remain host work. Parsing produces descriptors and raw
input data; it must not conceal CPU evaluation over mesh cells or samples.
Standalone NF2FF/SAR post-processing and time-stepping optimizations are separate
work; their existing interfaces and results must continue to work.

"Complete" means all supported built-in features have a GPU implementation,
not that arbitrary third-party C++ callbacks can run on a GPU. Custom primitives,
properties and extensions need explicit device adapters. Missing adapters must
be diagnosed before construction; strict mode must never silently use them on
the CPU. Metal construction is outside this HIP plan.

Keep `gpu_setup='cpu'|'auto'|'gpu'` and the existing CLI spelling. Keep CPU as the
default during implementation. In `auto`, an unsupported feature or recoverable
resource shortage rebuilds a fresh CPU operator. Strict `gpu` requires the
implemented device path. Runtime device faults and invalid numerical results
remain errors. A default change requires a separate measured decision.

## Coverage to finish

| Area | Baseline | Required result |
|---|---|---|
| Cartesian geometry | Box, Sphere, Cylinder, LinPoly; affine transforms | Preserve these and cover every remaining primitive below. |
| Imported solids | No Polyhedron/PolyhedronReader construction | Shared triangle representation, GPU spatial index and containment for STL/PLY and native polyhedra. |
| Other primitives | No MultiBox, Polygon, RotPoly, SphericalShell, CylindricalShell, Curve, Wire, Point or UserDefined | Match each primitive's CPU semantics, including special PEC curve paths and expression-defined geometry. |
| Materials | Constant component values; two averaging methods | Spatial weights, all built-in dispersion weights, discrete material databases, coordinate/transform rules and requested material/density samples. |
| Sources | Constant weights; one TFSF source | Spatial and file weights, device waveform generation and snapping; resolve multi-TFSF behavior explicitly. |
| Extensions | HIP arrays for UPML, excitation, RC/RLC, Mur/local absorbers, TFSF, Lorentz/Drude, Debye and conducting sheets | Complete their remaining numerical preparation, steady-state setup, and device ownership. |
| Coordinates | Cartesian setup only | Cylindrical input/primitive coordinates, cylindrical meshes, axis/seam rules and nested cylindrical multigrid. |
| Construction infrastructure | Host candidate lists, conservative scratch, host final timestep reduction | GPU candidate generation, scans/compaction, reductions and bounded memory planning. |
| Engine handoff | Arrays read back, packed on CPU and uploaded again | Direct device ownership transfer, GPU palettes/fusion metadata and initialization of fields/extension state. |
| Consumers | Host operator/extension arrays expected | Explicit materialization for debug/API consumers; GPU-generated probe/dump sampling tables and compatible processing. |

The primitive restrictions concern material, metal and excitation geometry.
Probe/output primitives must be inventoried separately so a point probe is not
mistaken for an unsupported solid.

## Design and ownership

1. Replace the growing fixed 48-double geometry record with typed, versioned
   descriptors and variable-length buffers for vertices, triangles, segments,
   expression programs and file data. Use stable primitive/property IDs and
   preserve CPU priority and equal-priority traversal order. Keep host CSXCAD
   pointers out of device descriptors.
2. Extend `GPU_OperatorSetup` with an owned construction result that
   `GPU_Backend_HIP` and its extension implementations can adopt. The setup object
   currently dies before engine initialization; move persistent allocations into
   this result rather than retaining dangling pointers or rebuilding arrays.
   Introduce this ownership contract early, then remove each host mirror as its
   consumers are migrated.
3. Separate immutable inputs, temporary workspaces and persistent outputs. Use
   checked host size arithmetic, explicit device index limits, count/scan/fill
   passes and allocation lifetimes. Budget the peak including stepping buffers,
   palettes, extensions and multigrid levels, not just construction scratch.
   Tile temporary work where possible; fitting setup alone does not prove the
   simulation fits. Preserve the existing runtime index limit unless a separate
   change updates and validates all stepping kernels too.
4. Keep FP64 geometry/material intermediates and existing float field/coefficient
   storage. Preserve mesh units, component orientation, averaging, PEC priority,
   source ordering and extension order. Do not introduce fast-math or approximate
   geometry as a shortcut to coverage.
5. Give built-in/custom adapters explicit preparation and construction support
   checks, including the matching engine extension. Provide per-phase activation,
   transfer and memory diagnostics. CPU and reference backends retain their
   existing construction route.

Primary implementation locations are [the setup interface](FDTD/gpu_operator_setup.h),
[model preparation](FDTD/operator_gpu_setup.cpp),
[HIP construction](FDTD/hip/operator_setup_hip.hip),
[HIP backend initialization](FDTD/hip/gpu_backend_hip.hip),
[GPU engine initialization](FDTD/engine_gpu.cpp), and the existing paired files in
`FDTD/extensions/`. Split new HIP geometry, expression and packing kernels into
focused files rather than extending one monolithic file indefinitely.

## Delivery sequence

Each milestone is a reviewable change or small series, with its own tests.
Temporary host mirrors may remain during migration; they do not satisfy the
final completion criteria. Geometry coverage starts with imported solids.

### 0. Reference contract and observable setup

- [ ] Inventory every CPU setup phase, primitive/property combination and
  extension, including host initialization after `CalcECOperator()`. Map each
  operation to a device implementation or the explicit host boundary above.
- [ ] Extend the construction tests with snapshots of material IDs, used flags,
  PEC edges, EC/coefficient arrays, timestep and extension tables. Tests must
  identify which setup phases actually executed on HIP.
- [ ] Record phase timing, construction H2D/D2H bytes, host/GPU peak memory,
  backend/device/compiler versions and source revisions. Existing engine transfer
  counters do not cover every construction or processing transfer.
- [ ] Introduce immutable descriptors and the owned result interface, initially
  retaining the current route as a comparison path.

Acceptance: reproducible CPU-setup/HIP-stepping and HIP-setup/HIP-stepping
baselines; a strict run cannot pass by falling back or skipping an extension.

### 1. Polyhedron and PolyhedronReader

- [ ] Export loaded vertices, accepted faces, dimensions, transforms and
  primitive priorities. Reuse the existing STL/PLY readers; both primitive types
  feed one geometry representation. Do not use `ToPolyhedron()` for the reader:
  its enum differs even though it inherits `CSPrimPolyhedron`.
- [ ] Build triangle bounds and a bounded traversal spatial index on HIP. Start
  with closed triangular solids, then cover all cases with defined CPU behavior.
  Query containment on the device at the actual material/PEC sampling positions.
- [ ] Specify boundary/intersection handling before choosing a ray test. The CPU
  implementation counts triangle intersections toward a random endpoint; faces
  with more than three vertices already carry a CPU warning. Establish repeatable
  oracle fixtures and resolve any demonstrated CPU ambiguity in a separate
  CSXCAD correctness change. Do not hide topology differences with field tolerances
  or introduce GPU-only mesh repair/triangulation semantics.
- [ ] Test native and imported solids: convex/concave, cavities and disconnected
  components where supported, reversed winding, translated negative coordinates,
  transforms, overlapping priorities, face/edge/vertex samples, and malformed or
  open input. Verify dimension/invalid-face behavior as well as interior points.

Acceptance: strict HIP setup for equivalent native, STL and PLY fixtures agrees
with an independently checked geometry oracle and CPU construction; large mesh
tests demonstrate bounded memory and avoid a full triangle scan per cell.

### 2. Remaining explicit geometry and transforms

- [ ] Add MultiBox, planar Polygon, RotPoly, SphericalShell, CylindricalShell,
  finite-radius Wire and Point using their existing query/dimension semantics.
  Preserve MultiBox priority as one primitive, shell boundaries, angular wrapping,
  wire endpoints and degenerate cases. A zero-volume primitive must not acquire
  a new artificial volume.
- [ ] Port metal Curve discretization through `FindPath()`/`CalcPEC_Curves()`:
  mesh snapping, segment traversal, PEC writes, used flags and counts. Curve
  containment always returns false on CPU, so a containment kernel alone cannot
  implement it. Test segment junctions, overlaps and out-of-domain endpoints.
- [ ] Support coordinate conversion and the full matrix behavior actually
  accepted by CSXCAD. Its transform applies the first three matrix rows without
  perspective division; do not interpret non-affine input as a new projective
  geometry feature. Audit inverse/bounds behavior and reject invalid transforms
  consistently.

Acceptance: focused geometry/PEC tests for every explicit primitive plus mixed
models; Cartesian meshes can use supported cylindrical-coordinate primitives.
UserDefined follows with the expression evaluator in milestone 3.

### 3. Expressions and file-backed data

- [ ] Compile expressions on the host into a documented device program and
  evaluate samples on HIP. Audit the installed fparser export surface first;
  avoid undocumented bytecode assumptions or a second, subtly different parser.
  Any required export API belongs in its owning repository. Add no dependency.
- [ ] Cover the grammar/functions exposed by CSXCAD and waveform expressions:
  arithmetic, comparisons, branching, constants/parameters, coordinate variables,
  time and CSFunctionParser's Bessel functions. Test domain errors, non-finite
  values, conditional evaluation and bounded evaluation storage. Arbitrary host
  callbacks require an explicit adapter, not a device function-pointer cast.
- [ ] Apply the evaluator to epsilon/mu/conductivity and dispersion weights,
  source weights, UserDefined geometry and custom waveforms. Preserve coordinate
  conversion, weight origins and per-component/per-pole behavior. Constant
  expressions such as `1/2` must work as well as numeric literals.
- [ ] Export immutable discrete-material mesh/index/database data after HDF5
  decoding. Port lookup, scale/transform order, database background selection,
  missing-channel fallback and density sampling; do not replace voxel lookup
  with interpolation. Test exact grid limits and out-of-domain coordinates.
- [ ] Export file-weighted source arrays and port the current transverse
  interpolation, propagation/component selection, origins and boundary handling.
  Test both file and expression paths against the same CPU definitions.

Acceptance: independent evaluator/lookup comparisons plus strict setup runs for
graded, dispersive, voxel and file-excitation models. Tiny test data is generated
or checked in with provenance; external user files are not required by CI.

### 4. Complete source and extension preparation

- [ ] Move Gaussian, sinusoidal, Dirac, step and custom waveform sample creation,
  source/RC/RLC snapping, UPML grading and remaining TFSF numerical metadata to
  HIP. Preserve waveform lengths, delays, periodic timestep adjustment, hard/soft
  E/H sources, lumped overlap order, caps and invalid-element behavior.
- [ ] Construct steady-state sampling/state and every existing extension's
  persistent tables on the device. Keep registration and small scalar reporting
  on the host. Preserve the extension execution order and existing fused paths.
- [ ] Resolve multi-TFSF separately: the CPU implementation currently returns
  after the first accepted plane-wave source. Removing the HIP preflight limit
  would not implement multiple sources. First test/preserve existing selection
  behavior; then implement explicit multiple-source storage and superposition in
  the paired operator/engine extensions and CPU/reference implementations before
  enabling it in HIP. Preserve single-source compatibility and test overlapping
  boxes, direction, polarization and delays. Keep other engines consistent.
- [ ] Replace the exact-type allowlist with explicit adapters whose supported
  semantics are checked. A subclass does not inherit GPU compatibility merely
  because a `dynamic_cast` succeeds.

Acceptance: every built-in extension completes its numerical preparation on HIP;
multi-source behavior has a validated shared reference. Existing extension and
analytical field comparisons remain within their established limits.

### 5. Cylindrical setup

- [ ] Share preparation/storage infrastructure with `Operator_Cylinder` instead
  of changing only the Cartesian gate in `openems.cpp`.
- [ ] Port radial/angular metrics, Yee coordinates, material averaging, boundary
  coefficients, axis corrections and closed-cylinder seam/ghost-line handling.
  Preserve the conservative timestep choice when the mesh contains `r=0`.
- [ ] Apply every CPU-supported extension and geometry combination in cylindrical
  coordinates; explicitly retain CPU restrictions for combinations it rejects.
  Test open angular sectors, full cylinders, meshes with/without the axis,
  nonuniform spacing and transformed/weighted materials.

Acceptance: strict cylindrical HIP construction passes coefficient checks and
the existing cylindrical/coaxial analytical and engine comparisons.

### 6. Cylindrical multigrid setup

- [ ] Build level descriptors, split positions, child/parent mappings,
  interpolation coefficients and missing material storage on HIP. Preserve
  extension cloning/ownership and the child-first timestep dependency, including
  rebuilds when the shared timestep changes.
- [ ] Use the existing shared HIP context/stream across levels; validate all
  levels and their combined peak memory before publishing the result.
- [ ] Transfer persistent construction data directly to the multigrid engine.
  Propagate failures from every child and release the whole partial hierarchy.

Acceptance: two-level and nested cases, nonuniform angular meshes, interface
sources/fields, compatible dispersive/PML combinations and forced resource
failures agree with CPU setup and leave the solver reusable.

### 7. Finish device ownership and remove bulk CPU setup

- [ ] Generate geometry bounds, acceleration structures and stable candidate
  lists on HIP; finish device reductions for the timestep and statistics.
  Avoid unconditional CPU CGAL-tree construction just to export imported data:
  separate decoded geometry from lazy CPU query acceleration in CSXCAD while
  preserving its existing public query behavior.
- [ ] Replace conservative dense extension scratch with measured count/scan/fill
  allocations and reusable workspaces. Check counts/offsets before allocation
  and kernel dispatch, including expressions, triangle indexes and all levels.
- [ ] Build core and extension palettes, index-width selection, dense alternatives
  and fusion metadata on HIP. Preserve coefficient values and stepping behavior;
  palette numbering may differ if decoding is identical. Avoid a new library
  dependency for sort/scan and retain the current compression decision contract.
- [ ] Adopt final buffers in `GPU_Backend_HIP::Init()` and device extension
  creation. Remove routine EC/material/coefficient/extension readbacks, host
  extraction/repacking and reuploads. Initialize fields and extension histories
  on HIP; eliminate unconditional allocation/zeroing of full host mirrors.
- [ ] Audit remaining consumers in `Common/`, engine interfaces and processing
  setup. Build numerical probe/dump/NF2FF sampling maps and needed material/density
  samples on HIP. Provide explicit, lazy host materialization for debug dumps,
  public getters or CPU processing that actually requests values. Do not let
  one getter cause an unreported full operator download during ordinary setup.

Acceptance: transfer traces from model upload through first timestep contain no
bulk construction-output round trip or CPU mesh/sample sweep in normal strict
HIP runs. Only input uploads, bounded status/scalar metadata and explicitly
requested host results remain. Debug/output paths still produce matching data.

### 8. Cross-platform validation and release

- [ ] Run the completed coverage matrix on NVIDIA HIP/CUDA and AMD HIP/ROCm;
  build CPU-only configurations and retain Metal/reference compatibility.
  Validate Windows MSVC and Linux builds without adding dependencies.
- [ ] Preserve the release's multiple NVIDIA architecture targets. Check the
  actual device images and run on representative older hardware, including a
  P5000 when available, as well as newer NVIDIA and AMD hardware. Compiled target
  coverage is not hardware validation; record unavailable combinations as untested.
- [ ] Update README, changelog, CLI help, Python and Octave/Matlab help together
  when behavior lands. Document support, fallback reasons, memory limits and
  intentional host work. Preserve the current normal release contents and wheels.
- [ ] Build a separate release, validate the extracted binaries/wheels and source
  archive, and record revisions, toolchains, devices, tests and hashes. Preserve
  existing builds and release archives; check disk space before a fresh build.

Acceptance: all required numerical/behavioral gates pass on recorded platforms,
and release claims distinguish compiled, tested and still-unavailable hardware.

## Validation and completion gates

For every feature, compare CPU setup plus HIP stepping against strict HIP setup
plus the same HIP stepping. Also retain basic/reference comparisons and analytical
tests so a shared stepping error cannot make two wrong setups appear correct.

- Require exact discrete IDs, topology decisions, indices, priorities, PEC masks
  and activation where the semantics are defined. Define and test geometry
  boundary rules explicitly. Compare numeric coefficients, expressions and
  timestep with documented scale-aware tolerances; set those tolerances from the
  reference/precision requirements before accepting results.
- Keep the existing per-file field limits: `1e-4` peak-relative and `1e-3`
  relative-L2, alongside finite-value checks, matching output inventories, meshes,
  sampling times/frequencies and analytical physical limits. Do not loosen these
  automatically to accommodate a new kernel.
- Exercise strict rejection, automatic fallback, malformed inputs, low-memory
  preparation, runtime failures, cancellation, debug dumps and repeated runs after
  both success and failure. Publish no partial buffers or changed primitive flags
  before successful construction; retry only from a clean operator at explicitly
  safe resource-failure boundaries.
- Extend `python/Tests/GPU_Setup.py` and run `GPU_Engine.py --require-backend HIP`,
  Python unit tests, CSXCAD tests for changed contracts, and the official Python
  and Octave smoketests. Add Octave fixtures for exposed behavior; the current
  `--all-engines` sweep does not itself select HIP. Report unavailable tools as
  untested rather than passed.
- Benchmark small analytic, many-primitive, large STL/PLY, weighted/voxel and
  cylindrical/multigrid cases. Record repeated cold/warm setup and total wall time,
  phase times, transfer volume and host/GPU memory against identical CPU inputs.
  Separate diagnostic/kernel-timed runs from throughput runs. Investigate
  regressions; report measured results without promising a fixed speedup.

The work is complete when the built-in coverage matrix passes in strict HIP,
normal setup has no hidden bulk CPU numerical path or construction round trip,
all consumers and fallback/lifetime tests pass, and the validated release records
its remaining hardware/resource limits. An unsupported built-in feature remains
an open milestone, even if `auto` successfully falls back.

## Repository boundaries and references

Solver adapters, kernels, engine handoff and solver tests belong in openEMS.
Read-only geometry/material exports and shared geometry semantics belong in
CSXCAD; read its `AGENTS.md` before editing it. Parser changes, if needed, belong
in fparser and must preserve its LGPL licensing and compatibility. No dependency
is added by this plan. Agree the core ownership/interface changes in milestone 0
before implementing them, as required by the solver's contribution rules.

Each implementation commit carries its required AI/DCO trailers. Commit submodule
work in its own repository; push it before a separate parent pointer update.
Build artifacts, validation data and machine-specific configuration stay ignored.

The pinned Vulkan reference `31f81b810ea36c2739b2c0096578b35c8f83e208`
already informed the current HIP construction. Its setup dispatcher also only
accepts Box, Sphere, Cylinder and LinPoly; it is a reuse source for established
algorithms, not evidence that the missing geometry has already been implemented.
Use the [current README](README.md), [CPU operator](FDTD/operator.cpp),
[cylindrical operator](FDTD/operator_cylinder.cpp),
[multigrid operator](FDTD/operator_cylindermultigrid.cpp),
[construction tests](python/Tests/GPU_Setup.py), and
[project documentation](https://docs.openems.de/en/latest/) as the maintained
references. Historical build snapshots remain unchanged.
