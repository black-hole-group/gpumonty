Adding GPU acceleration to `grmonty`: `gpumonty`
==============================================

**Broken but could be fun to fix!**

Here are efforts at creating a GPU-accelerated version of grmonty with CUDA support.

# Important branches

The important branches for this project are the following:

- `acc2cuda`: contains all efforts at porting the alpha OpenACC version to CUDA. This is the most important branch regarding GPU acceleration.
- `openacc`: branch original em openacc, desatualizada
- `deterministic`: testes para tornar o resultado determinístico, não é tão importante
- `tester`: adicionar conjunto de testes. merge master

arquivos mais importantes
grmonty.cu


onde paramos na última vez
- corretude do código: variável max_tau_scatt atualizada o tempo todo durante execução na versão original, torna resultados não-determinístico [dado o mesmo seed, não temos as mesmas saídas]
- performance do código: como agrupar fótons com mesmo nstep [ML]
- trocar openmp => pthreads


precisaria de mais refatorização: mais organização do código


# Pseudocodes

A set of python codes written only for educational purposes, for understanding what the code does. Includes a proposal for a GPU version.

- `pseudocode/cpu.py`: basic steps that the current version of the codes perfoms
- `pseudocode/gpu.py`: a proposal for a GPU version
- `pseudocode/mpi.py`: a proposal for a MPI version