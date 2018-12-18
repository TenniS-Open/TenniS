#!/usr/bin/env python

"""
Author: Kier
"""

import struct


def write_int(stream, i):
    # type: (file, int) -> None
    stream.write(struct.pack("=i", i))


def read_int(stream):
    # type: (file) -> int
    return int(struct.unpack('=i', stream.read(4))[0])


def write_int_list(stream, a):
    # type: (file, Union[list[int], tuple[int]]) -> None
    if isinstance(a, list) or isinstance(a, tuple):
        stream.write(struct.pack("=i", len(a)))
        for size in a:
            stream.write(struct.pack("=i", size))
    else:
        raise Exception("The shape must be list or tuple")


def read_int_list(stream):
    # type: (file) -> list[int]
    shape_size = struct.unpack('=i', stream.read(4))[0]
    a = []
    for i in range(shape_size):
        size = struct.unpack('=i', stream.read(4))[0]
        a.append(size)
    return a
