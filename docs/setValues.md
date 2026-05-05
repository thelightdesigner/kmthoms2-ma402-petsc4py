
    Insert or add values into certain locations of the matrix.

    This is the primary method for populating the Jacobian matrix entries 
    calculated in the `formJacobian` routine.

    Parameters
    ----------
    rows : array_like (int)
        Local row indices where values are inserted.
    cols : array_like (int)
        Local column indices where values are inserted.
    values : array_like (scalar)
        A block of values (e.g., a 2D nested list or NumPy array) of 
        shape (len(rows), len(cols)).
    addv : int, optional
        Whether to insert (`PETSc.InsertMode.INSERT_VALUES`) or 
        add (`PETSc.InsertMode.ADD_VALUES`) to existing entries.

    Notes
    -----
    PETSc translates the Python list/array into a contiguous C array of 
    `PetscScalar` types[cite: 101]. After calling this, you MUST call 
    `assemble()` to begin and end the communication of matrix values.

    Example
    -------
    >>> J.setValues([0, 1], [0, 1], [[2.0, 1.0], [1.0, 2.0]])
    >>> J.assemble()