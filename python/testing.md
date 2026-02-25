## Testing TLDR

The testing script `run_test.sh` compares your SED against a reference SED generated with over 1E8 superphotons. It simply computes a cumulative log difference and prints the difference to stdout, along with a plot in `output/spectrum_scattering_comparison.png`. 

Here is a caveat: if you are generating a SED with small number of superphotos, let's say 1E6, your model will be affected by Poisson statistics and we have to take these effects into account, especially and the low and high energy ends of the spectrum. The test does not currently take that into account.

## TODO

- [ ] Incorporate proper Poisson statistics in the test