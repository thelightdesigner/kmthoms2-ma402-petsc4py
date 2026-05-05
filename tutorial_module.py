import sys
import math
import petsc4py

# Initialize PETSc with command-line arguments before importing PETSc
petsc4py.init(sys.argv)
from petsc4py import PETSc


class SimpleSystem:
    """Evaluates F(x) and the Jacobian for the default 2-variable system."""
    
    def formFunction(self, snes, X, F):
        # Obtain references to the underlying arrays
        x = X.getArray(readonly=True)
        f = F.getArray()
        
        # Compute function: F(x)
        f[0] = x[0]**2 + x[0]*x[1] - 3.0
        f[1] = x[0]*x[1] + x[1]**2 - 6.0

    def formJacobian(self, snes, X, J, P):
        x = X.getArray(readonly=True)
        
        # Define the 2x2 Jacobian matrix entries
        val = [
            [2.0 * x[0] + x[1], x[0]],
            [x[1],              x[0] + 2.0 * x[1]]
        ]
        
        # Insert values into the matrix and assemble
        J.setValues([0, 1], [0, 1], val)
        J.assemble()
        
        # If the preconditioner matrix P is different, assemble it too
        if J != P:
            P.assemble()
            
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


class HardSystem:
    """Evaluates F(x) and the Jacobian for the '-hard' 2-variable system."""
    
    def formFunction(self, snes, X, F):
        x = X.getArray(readonly=True)
        f = F.getArray()
        
        f[0] = math.sin(3.0 * x[0]) + x[0]
        f[1] = x[1]

    def formJacobian(self, snes, X, J, P):
        x = X.getArray(readonly=True)
        
        val = [
            [3.0 * math.cos(3.0 * x[0]) + 1.0, 0.0],
            [0.0,                              1.0]
        ]
        J.setValues([0, 1], [0, 1], val)
        J.assemble()
        
        if J != P:
            P.assemble()
            
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


def main():
    # Parse command line options to check for the -hard flag
    is_hard = PETSc.Options().getBool('hard', False)

    # -------------------------------------------------------------------------
    # 1. Create and configure the nonlinear solver context
    # -------------------------------------------------------------------------
    snes = PETSc.SNES().create()
    snes.setType(PETSc.SNES.Type.NEWTONLS)
    snes.setOptionsPrefix("mysolver_")

    # -------------------------------------------------------------------------
    # 2. Create matrix and vector data structures
    # -------------------------------------------------------------------------
    x = PETSc.Vec().createSeq(2)
    r = x.duplicate()

    # Create dense Jacobian matrix (2x2)
    J = PETSc.Mat().createDense(size=[2, 2])
    J.setUp()

    # -------------------------------------------------------------------------
    # 3. Assign routines based on the selected system
    # -------------------------------------------------------------------------
    if not is_hard:
        system = SimpleSystem()
    else:
        system = HardSystem()

    snes.setFunction(system.formFunction, r)
    snes.setJacobian(system.formJacobian, J)

    # -------------------------------------------------------------------------
    # 4. Customize linear solver (KSP) and preconditioner (PC) context
    # -------------------------------------------------------------------------
    ksp = snes.getKSP()
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.NONE)
    ksp.setTolerances(rtol=1.e-4, max_it=20)

    # Allow runtime options (like -snes_monitor) to override hardcoded defaults
    snes.setFromOptions()

    # -------------------------------------------------------------------------
    # 5. Evaluate initial guess and solve
    # -------------------------------------------------------------------------
    if not is_hard:
        x.set(0.5)
    else:
        x.setArray([2.0, 3.0])

    snes.solve(None, x)

    # -------------------------------------------------------------------------
    # 6. Output Results and Cleanup
    # -------------------------------------------------------------------------
    print(f"Converged Reason: {snes.getConvergedReason()}")
    print(f"Number of Iterations: {snes.getIterationNumber()}")
    print("Solution Vector:")
    x.view()

    # While petsc4py utilizes Python's garbage collection, explicitly 
    # calling destroy() creates a direct analog to the C source mapping
    x.destroy()
    r.destroy()
    J.destroy()
    snes.destroy()


if __name__ == '__main__':
    main()