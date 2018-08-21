# Grmonty Tester

`tester.py` performs a end-to-end test upon grmonty, testing correcteness of the resulting spectrum and the following values (here referenced as 'output info'):
- luminosity
- dMact
- efficiency
- L/Ladv
- max_tau_scatt
- N_superph_made
- N_superph_recorded

# Quick start
To run the tester (running from tests/..):

    ./tests/tester.py

To run the reference extractor (running from tests/..):

    ./tests/extract_reference.py

The extractor updates tester's references. It will run grmonty with the desired parameters (see Settings for instructions on how to change these parameters), and rewrite `references.py` with the outputs extracted from these new executions.

# Settings
`tester.py` and `extract_references.py` are quite configurable.  To adjust their parameters, all you have to do is to modify the global constants at the top of each program's code.

The reference code output (used by `tester.py` to validate the testing code) is stored in `references.py`. This file can be changed manually (to add new references or change current ones), but its easier to update it with the `extract_reference.py` (see Quick Start on how to use it).

# TODO

[ ] Fix spectrum difference plot (plotted upon test failure)
[ ] Make options at command line (settings and option to save stderr and/or spectrum)
[ ] Make temporary dir to compile and run
[x] Read default reference values from separate file
[x] Perform more then one round for each test and take mean
[x] Allow higher limit for some values (especially max_tau_scatt)