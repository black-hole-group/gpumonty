Adding GPU acceleration to `grmonty`: `gpumonty`
==============================================

**Broken but could be fun to fix!**

Here are efforts at creating a GPU-accelerated version of grmonty with CUDA support.

# Important branches

The important branches for this project are the following:

- `acc2cuda`: contains all efforts at porting the alpha OpenACC version to CUDA. This is the most important branch regarding GPU acceleration.
- `openacc`: contains work towards an OpenACC version.
- `deterministic`: tests for obtaining a deterministic result. Not so important. 
- `tester`: added tests, merged with master.

The most important file in branch `acc2cuda` is `grmonty.cu`.

# Where did we stop?

**`max_tau_scatt`**. The main problem with the code is that the `max_tau_scatt` variable is continuously updated in the original version and shared among all threads. This makes the results non-deterministic in computer science lingo, in other words, given the same seed we do not get the same output. For a proper CUDA version, we cannot have variables being shared among threads --- this will destroy performance. 

How to group superphotons with the same `nstep`? This impacts code performance. 

**Refactoring**. Code needs to refactored and better organized. 


# Pseudocodes

A set of python codes written only for educational purposes, for understanding what the code does. Includes a proposal for a GPU version.

- `pseudocode/cpu.py`: basic steps that the current version of the codes perfoms
- `pseudocode/gpu.py`: a proposal for a GPU version
