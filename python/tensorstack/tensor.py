#!/usr/bin/env python

"""
:author Kier
"""

import numpy
import struct

from prototype import Prototype
from prototype import write_prototype
from prototype import read_prototype


def from_any(val, dtype=None):
    pass


def to_int(val):
    pass


def to_float():
    pass


def to_str():
    pass


def prod(shape):
    return numpy.prod(shape)


def write_tensor(stream, tensor):
    # type: (file, numpy.ndarray) -> None
    """
    Write numpy.ndarray to ts.tensor
    :param stream: a stream ready to write
    :param tensor: numpy.ndarray
    :return: None
    """
    proto = Prototype(dtype=tensor.dtype, shape=tensor.shape)
    write_prototype(stream=stream, proto=proto)
    bytes = tensor.newbyteorder('<').tobytes()
    assert proto.count * proto.dtype_bytes == len(bytes)
    stream.write(struct.pack("=%ds" % len(bytes), bytes))
    pass


def read_tensor(stream):
    # type: (file) -> numpy.ndarray
    """
    Read ts.tensor from stream
    :param stream: a stream contains ts's tensor
    :return:
    """
    proto = read_prototype(stream=stream)
    print(proto)
    bytes = stream.read(proto.count * proto.dtype_bytes)
    dtype = numpy.dtype(proto.dtype_numpy)
    dtype = dtype.newbyteorder('<')
    tensor = numpy.frombuffer(bytes, dtype=dtype)
    tensor = numpy.resize(tensor, new_shape=proto.shape)
    return tensor


if __name__ == '__main__':
    a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    tensor = numpy.asarray(a, dtype="float32")
    dtype = tensor.dtype
    shape = tensor.shape

    with open("tensor.txt", "wb") as fo:
        print("Write dtype={}, shape={}".format(dtype, shape))
        write_tensor(fo, tensor=tensor)

    with open("tensor.txt", "rb") as fi:
        local_tensor = read_tensor(fi)
        print("Read dtype={}, shape={}".format(local_tensor.dtype, local_tensor.shape))
        print(local_tensor)
    pass
