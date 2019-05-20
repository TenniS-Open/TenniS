#!/usr/bin/env python

"""
:author Kier
"""

import numpy
import struct
import sys

from .prototype import Prototype
from .prototype import write_prototype
from .prototype import read_prototype

from . import dtype as ts_dtype


def compatible_string(obj):
    # type: (object) -> object
    if sys.version > '3':
        pass
    else:
        if isinstance(obj, unicode):
            return str(obj)
    return obj


class StringTensor(object):
    def __init__(self, s):
        # type: (Union[str, bytes]) -> None
        s = compatible_string(s)
        if isinstance(s, str):
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


class PackedTensor(object):
    def __init__(self, fields=None):
        # type: (Union[numpy.ndarray]) -> None
        self.__fields = []
        self.pack(fields)

    def __getitem__(self, item):
        return self.__fields[item]

    def __setitem__(self, key, value):
        self.__fields[key] = value

    def __len__(self):
        return len(self.__fields)

    def pack(self, fields):
        if fields is None:
            self.__fields = []
            return

        if not isinstance(fields, list):
            pass
        elif not isinstance(fields, tuple):
            fields = list(fields)
        else:
            fields = [fields,]
        self.__fields = fields

    def unpack(self):
        return self.__fields

    def __iter__(self):
        return self.__fields.__iter__()

    @property
    def dtype(self):
        if len(self.__fields) == 0:
            return 'void'
        return self.__fields[0].dtype

    @property
    def shape(self):
        if len(self.__fields) == 0:
            return ()
        return self.__fields[0].shape

    def __str__(self):
        return "{%s}" % (", ".join([str(field) for field in self.__fields]))

    def __repr__(self):
        return "{%s}" % (", ".join([repr(field) for field in self.__fields]))


__map_dtype_to_numpy = {}


def register_dtype(dtype, to_numpy):
    __map_dtype_to_numpy[dtype] = to_numpy


def from_any(val, dtype=None):
    # type: (Union[numpy.ndarray, str, StringTensor], Union[numpy.dtype, int]) -> Union[numpy.ndarray, StringTensor]
    if isinstance(dtype, int):
        dtype = ts_dtype.to_numpy(dtype=dtype)

    if isinstance(val, StringTensor):
        return val

    if isinstance(val, PackedTensor):
        return PackedTensor([from_any(o, dtype=dtype) for o in val.unpack()])

    # compress str as StringTensor
    val = compatible_string(val)
    if isinstance(val, (str, bytes)):
        return StringTensor(val)

    # do not convert ndarray
    if dtype is None and isinstance(val, numpy.ndarray):
        return val

    # convert dtype
    elif type(val) in __map_dtype_to_numpy:
        val = __map_dtype_to_numpy[type(val)](val)

    return numpy.asarray(val, dtype=dtype)


def to_int(val):
    # type: (Union[StringTensor, numpy.ndarray]) -> int
    if isinstance(val, StringTensor):
        return int(str(val))
    if not isinstance(val, numpy.ndarray):
        raise Exception("Can not convert dtype={} to int".format(type(val)))
    if len(val.shape) == 0:
        return int(val)
    if len(val.shape) != 1 or val.shape[0] != 1:
        raise Exception("Can not convert shape={} to int".format(val.shape))
    return int(val[0])


def to_float(val):
    # type: (Union[StringTensor, numpy.ndarray]) -> float
    if isinstance(val, StringTensor):
        return float(str(val))
    if not isinstance(val, numpy.ndarray):
        raise Exception("Can not convert dtype={} to float".format(type(val)))
    if len(val.shape) == 0:
        return float(val)
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
    # type: (file, Union[str, bytes, StringTensor]) -> None
    s = compatible_string(s)
    if isinstance(s, StringTensor):
        s = s.tobytes()
    elif isinstance(s, str):
        s = s.encode()
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


def write_unpacked_tensor(stream, tensor):
    # type: (file, Union[numpy.ndarray, int, float, str, bytes, list, tuple, StringTensor]) -> None
    """
    Write numpy.ndarray to ts.tensor
    :param stream: a stream ready to write
    :param tensor: numpy.ndarray
    :return: None
    """
    # write special string type
    tensor = compatible_string(tensor)
    if isinstance(tensor, (str, bytes, StringTensor)):
        __write_string(stream=stream, s=tensor)
        return

    # check input type and write common tensor
    if isinstance(tensor, (int, float)):
        tensor = from_any(tensor)
    elif isinstance(tensor, (list, tuple)):
        tensor = from_any(tensor, dtype=numpy.float32)
    elif isinstance(tensor, numpy.ndarray):
        pass
    elif type(tensor) in __map_dtype_to_numpy:
        tensor = numpy.asarray(__map_dtype_to_numpy[type(tensor)](tensor))
    else:
        raise Exception("Can not write type={} as tensor. Should call register_dtype firstly.".format(type(tensor)))
    # 1. write prototype
    proto = Prototype(dtype=tensor.dtype, shape=tensor.shape)
    # 2. write memory
    write_prototype(stream=stream, proto=proto)
    if proto.dtype == ts_dtype.BOOLEAN:
        tensor = numpy.asarray(tensor, dtype=numpy.uint8)   # using uint8 instead boolean
    tensor_bytes = tensor.newbyteorder('<').tobytes()
    assert proto.count * proto.dtype_bytes == len(tensor_bytes)
    stream.write(struct.pack("=%ds" % len(tensor_bytes), tensor_bytes))


def read_unpacked_tensor(stream):
    # type: (file) -> Union[numpy.ndarray, StringTensor]
    """
    Read ts.tensor from stream
    :param stream: a stream contains ts's tensor
    :return:
    """
    # read special string type
    # 1. read prototype
    # read common string type
    proto = read_prototype(stream=stream)
    if proto.dtype == ts_dtype.CHAR8:
        s = __read_raw_string(stream=stream, n=proto.count)
        return s
    # 2. read memory
    bytes = stream.read(proto.count * proto.dtype_bytes)
    dtype_numpy = numpy.dtype(proto.dtype_numpy)
    dtype_numpy = dtype_numpy.newbyteorder('<')
    tensor = numpy.frombuffer(bytes, dtype=dtype_numpy)
    tensor = numpy.resize(tensor, new_shape=proto.shape)
    return tensor


def write_tensor(stream, tensor):
    # type: (file, Union[numpy.ndarray, int, float, str, bytes, list, tuple, StringTensor, PackedTensor]) -> None
    # a. write packed tensor
    if isinstance(tensor, PackedTensor):
        field_count = len(tensor)
        stream.write(struct.pack("=i", field_count))

        for field in tensor:
            write_unpacked_tensor(stream=stream, tensor=field)

        return

    # 0. write field_count
    field_count = 1
    stream.write(struct.pack("=i", field_count))

    write_unpacked_tensor(stream=stream, tensor=tensor)


def read_tensor(stream):
    # type: (file) -> Union[numpy.ndarray, StringTensor, PackedTensor]

    # 0. read field count
    field_count = struct.unpack("=i", stream.read(4))[0]
    if field_count == 1:
        return read_unpacked_tensor(stream=stream)

    fields = []
    for i in range(field_count):
        fields.append(read_unpacked_tensor(stream=stream))

    return PackedTensor(fields)


def write(path_or_stream, tensor):
    # type: (Union[file, str], Union[numpy.ndarray, int, float, str, bytes, list, tuple, StringTensor, PackedTensor]) -> None
    path_or_stream = compatible_string(path_or_stream)
    if isinstance(path_or_stream, str):
        with open(path_or_stream, "wb") as stream:
            write_tensor(stream, tensor)
    else:
        write_tensor(path_or_stream, tensor)


def read(path_or_stream):
    # type: (Union[file, str]) -> Union[numpy.ndarray, StringTensor, PackedTensor]
    path_or_stream = compatible_string(path_or_stream)
    if isinstance(path_or_stream, str):
        with open(path_or_stream, "rb") as stream:
            return read_tensor(stream)
    else:
        return read_tensor(path_or_stream)


if __name__ == '__main__':
    a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    #tensor = from_any("ABC")
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
