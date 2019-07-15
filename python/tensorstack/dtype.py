#!/usr/bin/env python

import numpy

VOID = 0
INT8 = 1
UINT8 = 2
INT16 = 3
UINT16 = 4
INT32 = 5
UINT32 = 6
INT64 = 7
UINT64 = 8
FLOAT16 = 9
FLOAT32 = 10
FLOAT64 = 11
PTR = 12
CHAR8 = 13
CHAR16 = 14
CHAR32 = 15
UNKNOWN8 = 16
UNKNOWN16 = 17
UNKNOWN32 = 18
UNKNOWN64 = 19
UNKNOWN128 = 20
BOOLEAN = 21
COMPLEX32 = 22
COMPLEX64 = 23
COMPLEX128 = 24

__dtype_bytes_map = {
    VOID: 0,
    INT8: 1,
    UINT8: 1,
    INT16: 2,
    UINT16: 2,
    INT32: 4,
    UINT32: 4,
    INT64: 8,
    UINT64: 8,
    FLOAT16: 2,
    FLOAT32: 4,
    FLOAT64: 8,
    CHAR8: 1,
    CHAR16: 2,
    CHAR32: 4,
    UNKNOWN8: 1,
    UNKNOWN16: 2,
    UNKNOWN32: 4,
    UNKNOWN64: 8,
    UNKNOWN128: 16,
    BOOLEAN: 1,
    COMPLEX32: 4,
    COMPLEX64: 8,
    COMPLEX128: 16,
}

__dtype_str_map = {
    VOID: 'void',
    INT8: 'int8',
    UINT8: 'uint8',
    INT16: 'int16',
    UINT16: 'uint16',
    INT32: 'int32',
    UINT32: 'uint32',
    INT64: 'int64',
    UINT64: 'uint64',
    FLOAT16: 'float16',
    FLOAT32: 'float32',
    FLOAT64: 'float64',
    PTR: 'ptr',
    CHAR8: 'char8',
    CHAR16: 'char16',
    CHAR32: 'char32',
    UNKNOWN8: 'unknown8',
    UNKNOWN16: 'unknown16',
    UNKNOWN32: 'unknown32',
    UNKNOWN64: 'unknown64',
    UNKNOWN128: 'unknown128',
    BOOLEAN: 'boolean',
    COMPLEX32: 'complex32',
    COMPLEX64: 'complex64',
    COMPLEX128: 'complex128',
}

convert_pairs = [
    (numpy.bool, BOOLEAN),
    (numpy.int8, INT8),
    (numpy.int16, INT16),
    (numpy.int32, INT32),
    (numpy.int64, INT64),
    (numpy.uint8, UINT8),
    (numpy.uint16, UINT16),
    (numpy.uint32, UINT32),
    (numpy.uint64, UINT64),
    (numpy.float16, FLOAT16),
    (numpy.float32, FLOAT32),
    (numpy.float64, FLOAT64),
    (numpy.complex64, COMPLEX64),
    (numpy.complex128, COMPLEX128),
]

__numpy_dtype_to_dtype = { pair[0]: pair[1] for pair in convert_pairs }
__dtype_to_numpy_dtype = { pair[1]: pair[0] for pair in convert_pairs }


def is_dtype(numpy_dtype, ts_dtype):
    # type: (numpy.dtype, int) -> bool
    """
    return if numpy's dtype is ts's dtype
    :param numpy_dtype: numpy.dtye
    :param ts_dtype: int value of ts.dtype
    :return: bool
    """
    if ts_dtype not in __dtype_to_numpy_dtype:
        return False
    return __dtype_to_numpy_dtype[ts_dtype] == numpy_dtype


def from_numpy(dtype):
    # type: (numpy.dtype) -> int
    """
    convert numpy.dtype to ts.dtype
    :param dtype: numpy.dtype
    :return: int value of ts.dtype
    """
    '''
    # Test in py2.7
    print(isinstance(dtype, numpy.float32)) --> False
    print(dtype is numpy.float32) --> False
    print(dtype == numpy.float32) --> True
    print(isinstance(type(dtype), numpy.float32)) --> False
    print(type(dtype) is numpy.float32) --> False
    print(type(dtype) == numpy.float32) --> False
    '''
    for pair in convert_pairs:
        if dtype == pair[0]:
            return pair[1]
    raise Exception("Not supported numpy.dtype={}".format(dtype))\


def to_numpy(dtype):
    # type: (int) -> numpy.dtype
    """
    convert ts.dtype to numpy.dtype
    :param dtype: int value of ts.dtype
    :return: numpy.dtype
    """
    if dtype not in __dtype_to_numpy_dtype:
        raise Exception("Not supported converting tensorstack dtype = {}".format(dtype))
    return __dtype_to_numpy_dtype[dtype]


def dtype_bytes(dtype):
    # type: (Union[int, numpy.dtype]) -> int
    if not isinstance(dtype, int):
        dtype = from_numpy(dtype=dtype)
    if dtype not in __dtype_bytes_map:
        raise Exception("Do not support dtype={}".format(dtype))
    return __dtype_bytes_map[dtype]


def dtype_str(dtype):
    # type: (Union[int, numpy.dtype]) -> str
    if not isinstance(dtype, int):
        dtype = from_numpy(dtype=dtype)
    if dtype not in __dtype_str_map:
        raise Exception("Do not support dtype={}".format(dtype))
    return __dtype_str_map[dtype]