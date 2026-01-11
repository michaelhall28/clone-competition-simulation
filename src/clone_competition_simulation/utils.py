import bisect


def find_ge(a, x):
    """Find leftmost item greater than or equal to x"""
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError
