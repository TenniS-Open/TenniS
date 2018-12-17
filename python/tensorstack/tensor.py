#!/usr/bin/env python

"""
:author Kier
"""

import numpy
import struct

from prototype import Prototype
from prototype import write_prototype
from prototype import read_prototype

import dtype as ts_dtype


class StringTensor(object):
    def __init__(self, s):
        if isinstance(s, unicode):
            s = str(s)
        elif isinstance(s, str):
            pass
        elif isinstance(s, bytes):
            s = str(s.decode())
        self.__s = s

    @property
    def dtype(self):
        return 'char8'

    @property
    def shape(self):
        return (len(self.__s), )

    def tobytes(self):
        return self.__s.encode()

    def __str__(self):
        return self.__s

    def __repr__(self):
        return '"{}"'.format(self.__s)


def from_any(val, dtype=None):
    # type: (Union[numpy.ndarray, str], Union[numpy.dtype, int]) -> Union[numpy.ndarray, StringTensor]
    if isinstance(dtype, int):
        dtype = ts_dtype.to_numpy(dtype=dtype)

    # compress str as StringTensor
    if isinstance(val, (str, unicode, bytes)):
        return StringTensor(val)

    return numpy.asarray(val, dtype=dtype)


def to_int(val):
    # type: (Union[StringTensor, numpy.ndarray]) -> int
    if isinstance(val, StringTensor):
        return int(str(val))
    if not isinstance(val, numpy.ndarray):
        raise Exception("Can not convert dtype={} to int".format(type(val)))
    if len(val.shape) != 1 or val.shape[0] != 1:
        raise Exception("Can not convert shape={} to int".format(val.shape))
    return int(val[0])


def to_float(val):
    # type: (Union[StringTensor, numpy.ndarray]) -> float
    if isinstance(val, StringTensor):
        return float(str(val))
    if not isinstance(val, numpy.ndarray):
        raise Exception("Can not convert dtype={} to float".format(type(val)))
    if len(val.shape) != 1 or val.shape[0] != 1:
        raise Exception("Can not convert shape={} to float".format(val.shape))
    return float(val[0])


def to_str(val):
    # type: (Union[StringTensor, numpy.ndarray]) -> str
    if isinstance(val, StringTensor):
        return str(val)
    raise Exception("Can not convert dtype={} to string".format(type(val)))


def prod(shape):
    return numpy.prod(shape)


def __write_string(stream, s):
    # type: (file, Union[str, unicode, bytes, StringTensor]) -> None
    if isinstance(s, StringTensor):
        s = s.tobytes()
    elif isinstance(s, str):
        s = s.encode()
    elif isinstance(s, unicode):
        s = str(s).encode()
    elif isinstance(s, bytes):
        pass
    else:
        raise Exception("Can not write type={} as string".format(type(s)))
    proto = Prototype(dtype=ts_dtype.CHAR8, shape=(len(s),))
    write_prototype(stream=stream, proto=proto)
    stream.write(struct.pack("=%ds" % len(s), s))


def __read_raw_string(stream, n):
    # type: (file, int) -> StringTensor
    bytes = stream.read(n)
    return StringTensor(bytes.decode())


def write_tensor(stream, tensor):
    # type: (file, Union[numpy.ndarray, int, float, str, unicode, bytes, list, tuple, StringTensor]) -> None
    """
    Write numpy.ndarray to ts.tensor
    :param stream: a stream ready to write
    :param tensor: numpy.ndarray
    :return: None
    """
    # write special string type
    if isinstance(tensor, (unicode, str, bytes, StringTensor)):
        __write_string(stream=stream, s=tensor)
        return

    # check input type and write common tensor
    if isinstance(tensor, (int, float)):
        tensor = from_any(tensor)
    elif isinstance(tensor, (list, tuple)):
        tensor = from_any(tensor, dtype=numpy.float32)
    elif isinstance(tensor, numpy.ndarray):
        pass
    else:
        raise Exception("Can not write type={} as tensor".format(type(tensor)))
    proto = Prototype(dtype=tensor.dtype, shape=tensor.shape)
    write_prototype(stream=stream, proto=proto)
    tensor_bytes = tensor.newbyteorder('<').tobytes()
    assert proto.count * proto.dtype_bytes == len(tensor_bytes)
    stream.write(struct.pack("=%ds" % len(tensor_bytes), tensor_bytes))


def read_tensor(stream):
    # type: (file) -> Union[numpy.ndarray, StringTensor]
    """
    Read ts.tensor from stream
    :param stream: a stream contains ts's tensor
    :return:
    """
    # read special string type
    proto = read_prototype(stream=stream)
    if proto.dtype == ts_dtype.CHAR8:
        s = __read_raw_string(stream=stream, n=proto.count)
        return s
    # read common string type
    bytes = stream.read(proto.count * proto.dtype_bytes)
    dtype_numpy = numpy.dtype(proto.dtype_numpy)
    dtype_numpy = dtype_numpy.newbyteorder('<')
    tensor = numpy.frombuffer(bytes, dtype=dtype_numpy)
    tensor = numpy.resize(tensor, new_shape=proto.shape)
    return tensor


if __name__ == '__main__':
    a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    tensor = from_any("ABC")
    #tensor = numpy.asarray(a, dtype="float32")
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
