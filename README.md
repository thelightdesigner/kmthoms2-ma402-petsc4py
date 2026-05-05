# kmthoms2-ma402-petsc4py

I decided to translate src/snes/tutorials/ex1.c to python. This tutorial demonstrates using Newton's method for a two-variable system.

The translation from ex1.c to petsc4py was performed using Google Gemini.  

## Challenges & Debugging
 - The one-shot failed because of naming errors and faulty configuration as expected.
 - The initial AI output hallucinated certain method names like createSeqDense and getArrayRead
 - I restructured the procedural C code into a more Pythonic class-based architecture (SimpleSystem and HardSystem) to manage the residual (formFunction) and Jacobian (formJacobian) callbacks.
  - Setting up the environment on WSL took a long time because of the PETSc C headers (3.24.4) with the petsc4py version to avoid compilation failures during wheel building.
