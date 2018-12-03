#!/usr/bin/env python

"""
:author Kier
"""

import numpy
import struct
import dtype
from dtype import from_numpy
from dtype import to_numpy


def write_int32_shape(stream, shape):
    if isinstance(shape, list) or isinstance(shape, tuple):
        stream.write(struct.pack("=i", len(shape)))
        for size in shape:
            stream.write(struct.pack("=i", size))
    else:
        raise Exception("The shape must be list or tuple")


def read_int32_shape(stream):
    shape_size = struct.unpack('=i', stream.read(4))[0]
    shape = []
    for i in range(shape_size):
        size = struct.unpack('=i', stream.read(4))[0]
        shape.append(size)
    return shape


def write_prototype(stream, dtype, shape):
    stream.write(struct.pack("=b", from_numpy(dtype=dtype)))
    write_int32_shape(stream=stream, shape=shape)


def read_prototype(stream):
    dtype_code = struct.unpack('=b', stream.read(1))[0]
    dtype = to_numpy(dtype=dtype_code)
    shape = read_int32_shape(stream=stream)
    return dtype, shape


def write_tensor(stream, tensor):
    # type: (file, numpy.ndarray) -> None
    """
    Write numpy.ndarray to ts.tensor
    :param stream: a stream ready to write
    :param tensor: numpy.ndarray
    :return: None
    """
    pass


def read_tensor(stream):
    # type: (file) -> numpy.ndarray
    """
    Read ts.tensor from stream
    :param stream: a stream contains ts's tensor
    :return:
    """
    pass


if __name__ == '__main__':
    a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    tensor = numpy.asarray(a, dtype="float32")
    dtype = tensor.dtype
    shape = tensor.shape

    with open("prototype.txt", "wb") as fo:
        print("Write dtype={}, shape={}".format(dtype, shape))
        write_prototype(stream=fo, dtype=dtype, shape=shape)

    with open("prototype.txt", "rb") as fi:
        local_dtype, local_shape = read_prototype(stream=fi)
        print("Read dtype={}, shape={}".format(local_dtype, local_shape))
        read_tensor = numpy.ones(shape=local_shape, dtype=local_dtype)
        print(read_tensor.dtype)


    pass
