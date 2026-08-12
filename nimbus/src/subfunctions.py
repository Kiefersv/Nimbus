""" Functionalities that deal with the coding of Nimbus """
import numpy as np

def aoftf(value):
    """
    AoFtF = array_or_function_to_function
    Nimbus is fully time-dependent and can therefore either take a static or
    time-dependent atmospheric structure. Internally, all these variables are handled
    as functions. Here, an input is checked if it is a function or array. In the latter
    case it is transformed into a function.

    Parameters
    ----------
    value : np.ndarray or function

    Return
    ------
    function
    """
    if callable(value):
        return value
    elif isinstance(value, np.ndarray):
        def aaf(p, t):
            """
            assigns each array layer but as a function

            Parameters
            ----------
            p : np.ndarray
                Pressure in cgs. This is a required dummy variable.
            t : np.ndarray
                Timestep in seconds. This is a required dummy variable.

            Return
            ------
            value : np.ndarray(len(p))
                mixing constant for each pressure
            """
            return value
        return aaf
    else:
        raise ValueError('Atmospheric structure inputs must be either a function '
                         'or array.')