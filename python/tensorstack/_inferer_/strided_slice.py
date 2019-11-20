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


def bit_ones(n):
    # type: (int) -> int
    count = 0
    while n != 0:
        count += 1
        n = (n - 1) & n
    return count


def resize(a, size, value=0):
    # type: (List[int], int, object) -> List[int]
    a.clear()
    a.extend([value, ] * size)
    return a


def infer_output_list(x, y, begin, end, stride, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, _in, _out):
    # type: (List[int], List[int], List[int], List[int], List[int], int, int, int, int, int, List[int], List[int]) -> bool
    if len(stride) == 0:
        resize(stride, len(begin), 1)
    if len(begin) != len(end) or len(begin) != len(stride):
        return False

    slice_size = len(begin)
    if slice_size == 0:
        return False

    ellipsis_ones = bit_ones(ellipsis_mask)
    if ellipsis_ones > 1:
        return False

    ellipsis_index = 0
    ellipsis_shape = []

    if ellipsis_ones:
        # first deal with in shape
        left_count = 0
        right_count = 0
        i = 0
        while i < slice_size:
            if ellipsis_mask & (1 << i):
                break
            left_count += 1
            if new_axis_mask & (1 << i):
                left_count -= 1
            i += 1
        ellipsis_index = i
        i += 1
        while i < slice_size:
            right_count += 1
            if new_axis_mask & (1 << i):
                right_count -= 1
            i += 1
        ellipsis_count = x.__len__() - left_count - right_count
        _in.clear()
        _in.extend(x[:left_count])
        _in.append(int(numpy.prod(x[left_count: left_count + ellipsis_count])))
        _in.extend(x[left_count + ellipsis_count:])
        stride[ellipsis_index] = 1
        begin_mask |= 1 << ellipsis_index
        end_mask |= 1 << ellipsis_index

        ellipsis_shape = x[left_count: left_count + ellipsis_count]
    else:
        _in[:] = x[:]

    # deal new_axis
    for i in range(slice_size):
        if new_axis_mask & (1 << i):
            if i > _in.__len__():
                return False
            _in.insert(i, 1)

    final_shape = []
    if slice_size < _in.__len__():
        final_shape = _in[slice_size:]
        final_size = int(numpy.prod(_in[slice_size:]))
        _in[slice_size:] = []
        _in.append(final_size)

        begin.append(0)
        end.append(0)
        stride.append(1)
        begin_mask |= 1 << slice_size
        end_mask |= 1 << slice_size

        slice_size += 1

    if _in.__len__() != slice_size:
        return False

    # calculate out
    resize(_out, slice_size, 0)
    for i in range(slice_size):
        _out[i],  begin[i], end[i] = infer_output(_in[i], begin[i], end[i], stride[i],
                                                  bool(begin_mask & (1 << i)), bool(end_mask & (1 << i)))
    y[:] = _out[:]

    # deal with may final shape
    if final_shape.__len__() != 0:
        _out.pop(-1)
        _out.extend(final_shape)
        slice_size -= 1

    # shrink output, and expand ellipsis_shape
    for i in range(slice_size - 1, -1, -1):
        if ellipsis_shape.__len__() != 0 and i == ellipsis_index:
            # expand ellipsis_index
            _out.pop(i)
            for e in ellipsis_shape[::-1]:
                _out.insert(i, e)
            continue
        if shrink_axis_mask & (1 << i):
            if _out[i] != 1:
                return False
            _out.pop(i)

    return True


def do_grid(shape):
    # type: (List[int]) -> Iterable
    anchors = numpy.meshgrid(*[range(i) for i in shape])
    anchors = [anchor.reshape([-1]) for anchor in anchors]
    count = anchors[0].shape[0]
    for i in range(count):
        yield tuple([anchor[i] for anchor in anchors])


def do_stride_slice(x, begin, end, stride, y):
    # type: (numpy.ndarray, List[int], List[int], List[int], numpy.ndarray) -> numpy.ndarray
    for anchor in do_grid(y.shape):
        dst = anchor
        anchor = list(anchor)
        for i in range(len(anchor)):
            anchor[i] = begin[i] + stride[i] * anchor[i]
        src = tuple(anchor)
        y[dst] = x[src]
    return y


def infer_stride_slice(x, shape, begin, end, stride, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    begin = list(begin)
    end = list(end)
    stride = list(stride)
    y = []
    _in = []
    _out = []

    succeed = infer_output_list(shape, y, begin, end, stride,
                                begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                                _in, _out)

    # print("x={}, y={}".format(shape, _out))

    if not succeed:
        return None, None

    if x is None:
        return None, _out

    # waiting return sliced value
    x = numpy.asarray(x)
    _value = numpy.zeros(y, dtype=x.dtype)
    _value = do_stride_slice(x.reshape(_in), begin, end, stride, _value).reshape(_out)

    return _value, _out