
    Set the function evaluation routine and the vector to store the residual.

    This method registers a Python callback that PETSc will call whenever it 
    needs to evaluate the nonlinear function F(x).

    Parameters
    ----------
    function : callable
        A function with the signature `f(snes, x, f)` where:
        - `snes` is the SNES context.
        - `x` is the current state vector (input).
        - `f` is the residual vector to be populated (output).
    f : PETSc.Vec, optional
        The vector where the residual F(x) will be stored. If None, PETSc 
        uses the internal residual vector.
    args : tuple, optional
        Extra positional arguments to pass to the function.
    kwargs : dict, optional
        Extra keyword arguments to pass to the function.

    Notes
    -----
    In the underlying C implementation, this routine sets the function 
    pointer used by `SNESComputeFunction`. The relative decrease in the 
    $L_2$ norm of this residual vector is often used as a convergence 
    criterion.

    Example
    -------
    >>> snes.setFunction(my_system.formFunction, residual_vec)