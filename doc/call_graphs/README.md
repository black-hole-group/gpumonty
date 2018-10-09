Contains files modified to use [`cflow2dot`](https://github.com/johnyf/pycflow2dot) to generate nice call graphs showing which functions call which.

# Commands

## All dependencies in one plot

	  cd grmonty
	  cat *c > call_graphs/all.c
    cflow2dot -i all.c -f png -x exclude.txt 

## Separate plots

	  cd grmonty
    cflow2dot -i *.c -f png -x call_graphs/exclude.txt -o call_graphs/


# Files

- `exclude.txt`: list of functions to be excluded

# Misc 

File `gprof.png` was created with the tools `gprof` and `gprof2dot`.