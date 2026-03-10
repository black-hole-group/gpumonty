# Geodesic Tracing Feature

This branch adds a **geodesic tracing mode** that traces superphoton geodesic paths through curved spacetime and saves Boyer-Lindquist coordinates (r, theta, phi) to an HDF5 file. This is useful for visualizing photon trajectories and debugging the geodesic integrator.

## Usage

Add the following to your `.par` file (set `Ns` to the desired number of traced photons):

```
Ns 1000
trace_geodesics 1
trace_stride 10
trace_maxsteps 10000
trace_output geodesics.h5
```

This generates ~1000 photons from the GRMHD emissivity distribution, traces them along null geodesics (no absorption/scattering), and saves BL coordinates every 10 steps to `output/geodesics.h5`.

When `trace_geodesics 1` is set, the normal spectrum computation is skipped entirely.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trace_geodesics` | 0 | Enable (1) or disable (0) tracing mode |
| `trace_stride` | 10 | Save position every N geodesic steps |
| `trace_maxsteps` | 10000 | Maximum geodesic steps per photon |
| `trace_output` | `geodesics.h5` | HDF5 output filename (written to `./output/`) |

The number of traced photons is controlled by `Ns` as usual.

## HDF5 Output Structure

| Dataset | Shape | Description |
|---------|-------|-------------|
| `r` | `[nph, max_saved]` | Boyer-Lindquist radial coordinate |
| `theta` | `[nph, max_saved]` | Boyer-Lindquist polar angle |
| `phi` | `[nph, max_saved]` | Azimuthal angle (Kerr-Schild convention, same as `X[3]`) |
| `nsteps` | `[nph]` | Actual number of saved steps per photon |
| `nph` | scalar | Number of traced photons |
| `max_saved_steps` | scalar | `trace_maxsteps / trace_stride + 1` |
| `trace_stride` | scalar | Stride used |
| `trace_maxsteps` | scalar | Max steps used |
| `Ns` | scalar | Input `Ns` parameter |

Photons that fall through the horizon or escape the outer boundary stop early; their remaining entries in the 2D arrays are zero. Use `nsteps[i]` to know how many entries are valid for photon `i`.

## Files Modified

| File | Change |
|------|--------|
| `src/decs.h` | Added trace fields to `Params` struct; added `of_trajectory` struct |
| `src/defs.h` | Same struct additions |
| `src/par.cu` | Default values, parsing, and reporting for trace parameters |
| `src/track.cu` | New `track_geodesic_save` device function |
| `src/track.h` | Declaration of `track_geodesic_save` |
| `src/kernels.cu` | New `track_geodesics` CUDA kernel; trace branch in `mainFlowControl` |
| `src/kernels.h` | Declaration of `track_geodesics` |
| `src/memory.cu` | `allocateTrajectoryData` and `freeTrajectoryData` |
| `src/memory.h` | Declarations of trajectory allocation functions |
| `src/main.cu` | `save_geodesics_h5` HDF5 output function; added `hdf5_utils.h` include |
| `src/main.h` | Declaration of `save_geodesics_h5` |
| `template.par` | Commented-out trace parameter examples |

## Design Notes

- **Simplified tracking**: No absorption, scattering, or weight roulette — pure geodesic integration via `push_photon` only.
- **Boundary check**: Uses `X[1] < startx[1]` (horizon) or `X[1] > stopx[1]` (escaped) directly, avoiding `stop_criterion` to prevent weight modification.
- **Memory layout**: Trajectory buffers are flat SoA arrays of size `nph * max_saved`, initialized to zero on the GPU. The `nsteps` array records how many entries are valid per photon.
- **Early return**: The trace branch returns from `mainFlowControl` immediately after saving, freeing all GPU resources before returning.
