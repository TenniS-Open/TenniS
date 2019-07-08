#!/usr/bin/env python

"""
:author Kier
"""

import numpy
import ctypes

TS_VOID        = 0
TS_INT8        = 1
TS_UINT8       = 2
TS_INT16       = 3
TS_UINT16      = 4
TS_INT32       = 5
TS_UINT32      = 6
TS_INT64       = 7
TS_UINT64      = 8
TS_FLOAT32     = 10
TS_FLOAT64     = 11
TS_CHAR8       = 13

"""
(ts_DTYPE, numpy.dtype, ctypes._SimpleCData)
"""
triple_list = [
    (TS_INT8, numpy.int8, ctypes.c_int8),
    (TS_INT16, numpy.int16, ctypes.c_int16),
    (TS_INT32, numpy.int32, ctypes.c_int32),
    (TS_INT64, numpy.int64, ctypes.c_int64),
    (TS_UINT8, numpy.uint8, ctypes.c_uint8),
    (TS_UINT16, numpy.uint16, ctypes.c_uint16),
    (TS_UINT32, numpy.uint32, ctypes.c_uint32),
    (TS_UINT64, numpy.uint64, ctypes.c_uint64),
    (TS_FLOAT32, numpy.float32, ctypes.c_float),
    (TS_FLOAT64, numpy.float64, ctypes.c_double),
    (TS_CHAR8, numpy.string_, ctypes.c_char),
]

map_key_ts_dtype = {triple[0]: (triple[1], triple[2]) for triple in triple_list}
map_key_numpy_dtype = {triple[1]: (triple[0], triple[2]) for triple in triple_list}
map_key_ctypes = {triple[2]: (triple[0], triple[1]) for triple in triple_list}


def query_map_tuple(m, k, i):
    if k not in m:
        return None
    return m[k][i]


def to_ts_dtype(dtype):
    # type: (Any) -> int
    result = None
    if isinstance(dtype, int):
        result = dtype
    elif isinstance(dtype, numpy.dtype):
        result = query_map_tuple(map_key_numpy_dtype, dtype.type, 0)
    elif isinstance(dtype, type):
        if issubclass(dtype, ctypes._SimpleCData):
            result = query_map_tuple(map_key_ctypes, dtype, 0)
        else:
            result = query_map_tuple(map_key_numpy_dtype, dtype, 0)

    if result is None:
        raise NotImplementedError("Can not checkout {} to ts_DTYPE".format(dtype))
    return result


def to_ctypes(dtype):
    # type: (Any) -> type
    result = None
    if isinstance(dtype, int):
        result = query_map_tuple(map_key_ts_dtype, dtype, 1)
    elif isinstance(dtype, numpy.dtype):
        result = query_map_tuple(map_key_numpy_dtype, dtype.type, 1)
    elif isinstance(dtype, type):
        if issubclass(dtype, ctypes._SimpleCData):
            result = dtype
        else:
            result = query_map_tuple(map_key_numpy_dtype, dtype, 1)
    if result is None:
        raise NotImplementedError("Can not checkout {} to ctypes".format(dtype))
    return result


def to_numpy_dtype(dtype):
    # type: (Any) -> type
    result = None
    if isinstance(dtype, int):
        result = query_map_tuple(map_key_ts_dtype, dtype, 0)
    elif isinstance(dtype, numpy.dtype):
        result = dtype.type
    elif isinstance(dtype, type):
        if issubclass(dtype, ctypes._SimpleCData):
            result = query_map_tuple(map_key_ctypes, dtype, 1)
        else:
            result = dtype
    if result is None:
        raise NotImplementedError("Can not checkout {} to ctypes".format(dtype))
    return result
