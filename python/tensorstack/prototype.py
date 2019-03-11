#!/usr/env/bin python

from .dtype import from_numpy
from .dtype import to_numpy
from .dtype import dtype_bytes
from .dtype import dtype_str

import numpy
import struct


class Prototype(object):
    def __init__(self, dtype, shape):
        if isinstance(dtype, int):
            self.__dtype = dtype
        else:
            self.__dtype = from_numpy(dtype=dtype)
        # len(shape = 0) means scalar
        # if len(shape) == 0:
        #     shape = (1, )
        self.__shape = shape
        self.__dtype_bytes = dtype_bytes(self.__dtype)

    def __str__(self):
        return "<dtype={}, shape={}>".format(dtype_str(self.__dtype), self.__shape)

    def __repr__(self):
        return self.__str__()

    @property
    def dtype(self):
        return self.__dtype

    @property
    def shape(self):
        return self.__shape

    @property
    def dtype_bytes(self):
        return self.__dtype_bytes

    @property
    def count(self):
        return numpy.prod(self.__shape, dtype=int)

    @property
    def dtype_numpy(self):
        return to_numpy(self.__dtype)


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


def __write_prototype_core(stream, dtype, shape):
    stream.write(struct.pack("=b", dtype))
    write_int32_shape(stream=stream, shape=shape)


def __read_prototype_core(stream):
    dtype = struct.unpack('=b', stream.read(1))[0]
    shape = read_int32_shape(stream=stream)
    return dtype, shape


def write_prototype(stream, proto):
    # type: (file, Prototype) -> None
    __write_prototype_core(stream=stream, dtype=proto.dtype, shape=proto.shape)


def read_prototype(stream):
    # type: (file) -> Prototype
    dtype, shape = __read_prototype_core(stream=stream)
    return Prototype(dtype=dtype, shape=shape)
