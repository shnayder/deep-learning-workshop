import numpy as np
import contextlib

def combine_histories(h1, h2):
    """
    Combine two history dicts -- appends the value lists to each other.

    Returns new dict.
    """
    d = {}
    for k in h1:
        d[k] = np.append(h1[k],h2[k])
    return d


# Some magic to make numpy arrays look prettier
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)
