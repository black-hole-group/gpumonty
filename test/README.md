# Grmonty Tester

`tester.py` performs a end-to-end test upon grmonty, testing correcteness of the resulting
spectrum and the following values (here referenced as 'output info'):

- luminosity
- dMact
- efficiency
- L/Ladv
- max_tau_scatt
- N_superph_made
- N_superph_recorded

# Quick start
To test the repo's code:

    ./tester.py

To extract output from the repo's code, in order to use as refererence for future tests:

    ./tester.py --extract

# Settings
`tester.py` is quite configurable.  To adjust it's parameters, all you have to do is to
modify the global constants at the top of the program's code. Each setting is well
documented at their definitions.

The reference code output (used by `tester.py` to validate the testing code) is stored in
`references.py`. This file can be changed manually (to add new references or change
current ones), but its easier to update it with the `./tester.py --extract`.

The `references.json` that comes in this repository was extracted from the original CPU
code (branch `master`) with scattering disabled.

# TODO

- [ ] Make cli options for global settings
- [ ] Make a setting to save the multiple executations' stderrs and/or spectrum (for debugging)
- [X] Make temporary dir to compile and run
- [X] Fix spectrum difference plot (plotted upon test failure)
- [X] Check when tester.py and extract_reference.py are called inside ./test and make them still work.
- [x] Read default reference values from separate file
- [x] Perform more then one round for each test and take mean
- [x] Allow higher limit for some values (especially max_tau_scatt)
