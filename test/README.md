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
To run the tester:

    ./tester.py

To run the reference extractor:

    ./extract_reference.py

The extractor updates tester's references. It will run grmonty with the desired parameters
(see Settings for instructions on how to change these parameters), and overwrite
`references.py` with the outputs extracted from these new executions.

# Settings
`tester.py` and `extract_references.py` are quite configurable.  To adjust their parameters,
all you have to do is to modify the global constants at the top of each program's code.

The reference code output (used by `tester.py` to validate the testing code) is stored in
`references.py`. This file can be changed manually (to add new references or change
current ones), but its easier to update it with the `extract_reference.py` (see Quick Start on
how to use it).

# TODO

- [ ] Make options at command line (settings and option to save the multiple executations' stderrs and/or spectrum)
- [X] Make temporary dir to compile and run
- [X] Fix spectrum difference plot (plotted upon test failure)
- [X] Check when tester.py and extract_reference.py are called inside ./test and make them still work.
- [x] Read default reference values from separate file
- [x] Perform more then one round for each test and take mean
- [x] Allow higher limit for some values (especially max_tau_scatt)
