import bisect

from numpy.typing import ArrayLike


def find_ge(a: ArrayLike, x: int | float) -> int:
    """Find index of leftmost item greater than or equal to x

    Parameters
    ----------
    a : ArrayLike
        Ordered array of values
    x : int | float
        Value

    Returns
    -------
    int
        index of the item found

    Raises
    ------
    ValueError
        _description_
    """
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError(f"Could not find item greater than or equal to {x}")
