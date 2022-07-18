`GRMONTY`: A relativistic Monte Carlo code
==========================================

Computes spectral energy distributions (SEDs) from numerical simulations of radiatively inefficient accretion flows around black holes. This version of GRMONTY is configured to use input files from the `HARM` code. It assumes that the source is a plasma near a black hole described by Kerr-Schild coordinates that radiates via thermal synchrotron and inverse compton scattering.

GRMONTY is parallelized using [OpenMP](https://en.wikipedia.org/wiki/OpenMP). For the alpha GPU version `gpumonty`, check out the [GPU documentation](GPU). This version is configured to use input from [`harm2d`](http://rainman.astro.illinois.edu/codelib/codes/ham2d/src/).

Pros:

- Deals with whatever curved spacetime as long as you provide the metric data
- Properly compute inverse Compton scattering in the Klein-Nishina regime

Cons:

- Only supports HARM data
- Only useful for post-processing radiative transfer (i.e. neglects momentum effect of photons on gas), thus only supports optically-thin flows
- Currently only supports multi-core CPUs and is inefficiently parallelized (see [GPU acceleration](GPU))
- Only handles a thermal electron distribution


Based on [Dolence et al. 2009 ApJ](http://adsabs.harvard.edu/abs/2009ApJS..184..387D). Originally downloaded from [Astrophysical Code Library](http://rainman.astro.illinois.edu/codelib/) @ [UI](http://illinois.edu).



# Quick start

Compile (requires OpenMP enabled `gcc`):

    make

Set number of threads (`csh`):

    setenv OMP_NUM_THREADS 8

If using `bash`:

    export OMP_NUM_THREADS=8

Run the code on the supplied harm output file:

    ./bin/grmonty 5000000 dump1000 4.e19

Arguments are:

- estimate of photon number (actual number is probabilistic due to scattering)
- harm dump file for model
- mass unit (few x 10^19 is appropriate for Sgr A*)

This will output spectrum to `spectrum.dat`.

## Plotting

Use python and the [`nmmn`](https://github.com/rsnemmen/nmmn) module:

```python
from  nmmn import plots
plots.plot('grmonty.spec')
```

## Running Tests

    ./test/tester.py

See ./test/README.md for test details.


## Calculate spectra from other sources

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

# Branch structure

See [`branches.md`](branches).


# References

- Code and methods: [Dolence et al. 2009 ApJ](http://adsabs.harvard.edu/abs/2009ApJS..184..387D)
- An early application: Sgr A\* SED model ([Moscibrodzka et al. 2009 ApJ](http://iopscience.iop.org/article/10.1088/0004-637X/706/1/497/meta)). A more recent Sgr A\* model by the same authors: [Moscibrodzka et al. 2014 A&A](http://www.aanda.org/articles/aa/abs/2014/10/aa24358-14/aa24358-14.html)
- More recent applications: M87 jet/RIAF SED ( [Moscibrodzka et al. 2016 A&A](http://www.aanda.org/articles/aa/abs/2016/02/aa26630-15/aa26630-15.html)), jet and RIAF SEDs for stellar mass black holes ([O'Riordan, Pe'er & McKinney 2016 ApJ](http://iopscience.iop.org/article/10.3847/0004-637X/819/2/95/meta))

# Citation

You are morally obligated to cite the following paper in any scientific literature that results from use of any part of GRMONTY:

> [Dolence, J.C., Gammie, C.F., Mo\'scibrodzka, M., \& Leung, P.-K. 2009, Astrophysical Journal Supplement, 184, 387]((http://adsabs.harvard.edu/abs/2009ApJS..184..387D))

If you use one of the alpha versions contained in this repo, for reproducibility purposes please cite this repo and the corresponding commit you use.

# TODO

- [x] make it work with [HARMPI](https://github.com/atchekho/harmpi)
- [ ] GPU support: CUDA, OpenACC
- [ ] add bremsstrahlung
- [ ] nonthermal electron distribution
- [ ] dynamic metrics as input
- [x] add LICENSE