#!/usr/bin/env python

"""
:author Kier
"""

import struct


class Header(object):
    MODULE_CODE_V1 = 0x19910929

    def __init__(self, fake=None, code=MODULE_CODE_V1):
        if fake is None:
            fake = 0
        if code is None:
            code = 0
        self.__fake = fake
        self.__code = code
        self.__data = bytes(('\0' * 120).encode())

    @property
    def fake(self):
        return self.__fake

    @fake.setter
    def fake(self, value):
        self.__fake = value

    @property
    def code(self):
        return self.__code

    @code.setter
    def code(self, value):
        self.__code = value

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value):
        assert isinstance(value, bytes) and len(value) == 120
        self.__data = value


def write_header(stream, header):
    # type: (file, Header) -> None
    if header.code == 0:
        raise Exception("Header's code must be set.")
    assert header.code == Header.MODULE_CODE_V1
    stream.write(struct.pack("=ii", header.fake, header.code))
    assert len(header.data) == 120
    stream.write(struct.pack("=%ds" % len(header.data), header.data))


def read_header(stream):
    # type: (file) -> Header
    fake = struct.unpack("=i", stream.read(4))[0]
    code = struct.unpack("=i", stream.read(4))[0]
    data_size = 120
    data = struct.unpack("=%ds" % data_size, stream.read(data_size))[0]
    header = Header(fake=fake, code=code)
    header.data = data
    return header
