Branch structure
==================

See `branches.md`.

- `master`, stable: matches the original release functionality, supports only input from `HARM2D`
- `illinois`: latest bug corrections by Gammie's group, `HARM2D`

Please note that all other branches include significant amount of work which has not been made public yet.

## 3D HARM support

work in progress...

## GPU acceleration

- `cuda`, in progress: CUDA version in progress, lead by Rodrigo
- `openacc`: OpenACC in progress, lead by Matheus

## Misc.

- `track_ph`: output photon world lines for visualization

# Pseudocodes

A set of python codes written only for educational purposes, for understanding what the code does. Includes a proposal for a GPU version.

- `pseudocode/cpu.py`: basic steps that the current version of the codes perfoms
- `pseudocode/gpu.py`: a proposal for a GPU version
- `pseudocode/mpi.py`: a proposal for a MPI version