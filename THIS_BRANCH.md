# Branch: `scattering-test`

This branch configures GPUmonty for the **scattering synchrotron sphere** validation test (Test 2 in CLAUDE.md) and adds the tooling to run and evaluate it.

## What is different from `main`

### Physics / model parameters (`src/sphere_model/model.h`)
- `NE_VALUE`: `1e13` → `1e18` cm⁻³ (optically thick regime that drives scattering)
- `THETAE_VALUE`: `100` → `3` (cooler electrons, so scattering dominates over thin synchrotron)
- `RMAX`: `10000` → `3100` cm (tighter outer boundary matching the test geometry)
- `N_ESAMP` / `N_EBINS`: `2500` → `5000` (finer spectral resolution)

### Sphere boundary treatment (`src/sphere_model/model.cu`)
- Replaced the hard binary inside/outside test with a **5×5 sub-cell anti-aliasing** scheme in `get_fluid_params`: each grid cell is sampled at 25 sub-points, and the plasma quantities (`Ne`, `B`, `Thetae`) are scaled by the fraction of sub-points that fall inside the sphere. This removes the staircase artefact at the sphere edge.
- `init_data` no longer fills the plasma array from a loop; fluid state is computed on the fly in `get_fluid_params` instead.
- Radial grid resolution raised from `N1 = 8192` to `N1 = 30000` to improve edge anti-aliasing quality.

### Build (`makefile`)
- Default model switched from `iharm_model` to `sphere_model`.
- Makefile guard added so GPU capability detection is skipped during `make clean`.
- Manual GPU CC fallback improved (was hard-coded `sm_86`; now re-uses the same variable logic).

### Parameter file (`test.par`)
- New parameter file for the scattering sphere run (high `ne`, low `Thetae`, scattering bias set accordingly).

### Testing infrastructure
- `run_test.sh`: shell script that builds the code, runs the simulation at multiple `Ns` values, and invokes the comparison script.
- `python/compare.py`: compares the new spectrum against the reference `igrmonty` output stored in `output/spectrum_scattering_igrmonty_1e8.h5`.
- `python/helper.py`: shared spectrum-loading utilities used by both the comparison script and the notebook.
- `python/sphere_example.ipynb`: updated notebook for visualising scattering sphere results.
- `output/sphere_scattering_*.spec`: pre-computed reference spectra at Ns = 1e5–1e8.
- `output/spectrum_scattering_igrmonty_1e8.h5`: `igrmonty` reference spectrum for direct comparison.
