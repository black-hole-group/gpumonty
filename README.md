GRMONTY: A relativistic Monte Carlo code
==========================================

Based on [Dolence et al. 2009 ApJ](http://adsabs.harvard.edu/abs/2009ApJS..184..387D). Originally downloaded from [Astrophysical Code Library](http://rainman.astro.illinois.edu/codelib/) @ [UI](http://illinois.edu).

This version of GRMONTY is configured to use input files from the HARM code available on the same site. It assumes that the source is a plasma near a black hole described by Kerr-Schild coordinates that radiates via thermal synchrotron and inverse compton scattering.

GRMONTY is parallelized using [OpenMP](https://en.wikipedia.org/wiki/OpenMP). This version is configured to use input from [`harm2d`](http://rainman.astro.illinois.edu/codelib/codes/ham2d/src/).

# Quick start

Compile (requires OpenMP enabled `gcc`):

    make

Set number of threads (`csh`):

    setenv OMP_NUM_THREADS 8

If using `bash`:

    export OMP_NUM_THREADS=8

Run the code on the supplied harm output file:

    ./grmonty 5000000 dump019 4.e19 [C]

Arguments are:

- estimate of photon number (actual number is probabilistic due to scattering)
- harm dump file for model
- mass unit (few x 10^19 is appropriate for Sgr A*)
- the integer C, if provided, will be the seed of the random number generator

This will output spectrum to `grmonty.spec` ~~which should be identical to `grmonty_spec_verify`~~.

# Plotting

Use python and the [`nmmn`](https://github.com/rsnemmen/nmmn) module:

```python
from  nmmn import plots
plots.plot('grmonty.spec')
```

# Calculate spectra from other sources

Replace `harm_model.c` with your own source model.  Begin by modifying `harm_model.c`. You must supply

```
init_model
make_super_photon
bias_func
get_fluid_params
report_spectrum
stop_criterion
record_criterion

gcon_func
gcov_func
get_connection
```

in the model file.

# Explanation of branches

- `master`, stable: matches the original release functionality, supports only input from `HARM2D`
- `illinois`: latest bug corrections by Gammie's group, `HARM2D`
- `progressbar`, stable: displays progress bar

## 3D HARMPI support

- `3dtests`, in progress, possibly unstable: new features being implemented, dump → ASCII
- `3d-stable`, stable: starting point for GPU version, dump → ASCII
- `read_harmpi`, buggy: trying to read `HARMPI` dumps directly in `grmonty`

## GPU support

- `cuda`, in progress: CUDA version
- `building-cuda`: CUDA progress by Caio

## Misc.

- `track_ph`: output photon world lines for visualization

# Pseudocodes

A set of python codes written only for educational purposes, for understanding what the code does. Includes a proposal for a GPU version.

- `pseudocode/cpu.py`: basic steps that the current version of the codes perfoms
- `pseudocode/gpu.py`: a proposal for a GPU version
- `pseudocode/mpi.py`: a proposal for a MPI version

# TODO

- [x] make it work with [HARMPI](https://github.com/atchekho/harmpi)
- [ ] GPU support: CUDA
- [ ] add bremsstrahlung
- [ ] nonthermal electron distribution
- [ ] dynamic metrics as input
- [x] add LICENSE

# References

- Code and methods: [Dolence et al. 2009 ApJ](http://adsabs.harvard.edu/abs/2009ApJS..184..387D)
- An early application: Sgr A\* SED model ([Moscibrodzka et al. 2009 ApJ](http://iopscience.iop.org/article/10.1088/0004-637X/706/1/497/meta)). A more recent Sgr A\* model by the same authors: [Moscibrodzka et al. 2014 A&A](http://www.aanda.org/articles/aa/abs/2014/10/aa24358-14/aa24358-14.html)
- More recent applications: M87 jet/RIAF SED ( [Moscibrodzka et al. 2016 A&A](http://www.aanda.org/articles/aa/abs/2016/02/aa26630-15/aa26630-15.html)), jet and RIAF SEDs for stellar mass black holes ([O'Riordan, Pe'er & McKinney 2016 ApJ](http://iopscience.iop.org/article/10.3847/0004-637X/819/2/95/meta))

# Citation

You are morally obligated to cite the following paper in any scientific literature that results from use of any part of GRMONTY:

> [Dolence, J.C., Gammie, C.F., Mo\'scibrodzka, M., \& Leung, P.-K. 2009, Astrophysical Journal Supplement, 184, 387]((http://adsabs.harvard.edu/abs/2009ApJS..184..387D))


# LICENSE

`grmonty` is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the `LICENSE` file or the [GNU General Public License](http://www.gnu.org/licenses/) for more details.
