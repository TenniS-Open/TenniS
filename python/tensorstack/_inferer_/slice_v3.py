from typing import Union, List, Tuple, Iterable
import sys
import numpy

import copy


def infer_output(x, begin, end, stride, begin_flag, end_flag):
    # type: (int, int, int, int, bool, bool) -> (int, int, int)
    """
    :param x:
    :param begin:
    :param end:
    :param stride:
    :param begin_flag:
    :param end_flag:
    :return: y, begin, end
    """
    # begin \in [0, x)
    # end \in [-1, x]
    if begin_flag:
        begin = 0 if stride > 0 else x - 1
    else:
        if stride > 0:
            if begin >= x:
                return 0, begin, end    # no elements
            elif begin < -x:
                begin = 0
            elif begin < 0:
                begin += x
        else:
            if begin < -x:
                return 0, begin, end    # no elements
            elif begin >= x:
                begin = x - 1
            elif begin < 0:
                begin += x

    if end_flag:
        end = x if stride > 0 else -1
    else:
        if stride > 0:
            if end <= -x:
                return 0, begin, end    # no elements
            elif end > x:
                end = x
            elif end < 0:
                end += x
        else:
            if end > x:
                return 0, begin, end    # no elements
            elif end <= -x:
                end = -1
            elif end < 0:
                end += x

    if stride > 0:
        return (end - begin - 1) // stride + 1 if begin < end else 0, begin, end
    elif stride < 0:
        return (begin - end - 1) // -stride + 1 if begin > end else 0, begin, end
    else:
        sys.stderr.write("slice step cant not be zero\n")
        return 0, begin, end


def _do_grid(shape):
    # type: (List[int]) -> Iterable
    anchors = numpy.meshgrid(*[range(i) for i in shape])
    anchors = [anchor.reshape([-1]) for anchor in anchors]
    count = anchors[0].shape[0]
    for i in range(count):
        yield tuple([anchor[i] for anchor in anchors])


def _do_slice(x, begin, end, stride, y):
    # type: (numpy.ndarray, List[int], List[int], List[int], numpy.ndarray) -> numpy.ndarray
    for anchor in _do_grid(y.shape):
        dst = anchor
        anchor = list(anchor)
        for i in range(len(anchor)):
            anchor[i] = begin[i] + stride[i] * anchor[i]
        src = tuple(anchor)
        y[dst] = x[src]
    return y


def infer(x, shape, starts, ends, axes=None, steps=None):
    # type: (numpy.ndarray, List[int], List[int], List[int], List[int], List[int]) -> (Union[numpy.ndarray, None], List[int])
    assert isinstance(shape, list)
    assert isinstance(starts, list)
    assert isinstance(ends, list)

    dims = len(shape)

    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)

    fixed_begins = [0] * dims
    fixed_ends = [0] * dims
    fixed_strides = [1] * dims
    fixed_begin_flags = [True] * dims
    fixed_end_flags = [True] * dims

    y_shape = list(shape)

    for i in range(len(axes)):
        axis = axes[i]
        start = starts[i]
        end = ends[i]
        step = steps[i]

        y_shape[axis], start, end = infer_output(shape[axis], start, end, step, False, False)

        fixed_begins[axis] = start
        fixed_ends[axis] = end
        fixed_strides[axis] = step
        fixed_begin_flags[axis] = False
        fixed_end_flags[axis] = False

    if x is None:
        return None, y_shape

    y = numpy.zeros(y_shape, x.dtype)

    _do_slice(x, fixed_begins, fixed_ends, fixed_strides, y)

    return y, y_shape
