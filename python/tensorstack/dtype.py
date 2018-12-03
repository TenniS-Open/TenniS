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
FLOAT64 = 12
PTR = 13
CHAR8 = 14
CHAR16 = 15
CHAR32 = 16
UNKNOWN8 = 17
UNKNOWN16 = 18
UNKNOWN32 = 19
UNKNOWN64 = 20
UNKNOWN128 = 21
BOOLEAN = 22
COMPLEX32 = 23
COMPLEX64 = 24
COMPLEX128 = 25

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
    (numpy.char, CHAR8),
]

__numpy_dtype_to_dtype = { pair[0]: pair[1] for pair in convert_pairs }
__dtype_to_numpy_dtype = { pair[1]: pair[0] for pair in convert_pairs }


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
        raise Exception("Not supported converting tensorstack dtype ={}".format(dtype))
    return __dtype_to_numpy_dtype[dtype]