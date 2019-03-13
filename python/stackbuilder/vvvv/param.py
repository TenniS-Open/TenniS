import struct


def Int(stream):
    # type: (Union[io.BinaryIO, io.BufferedReader]) -> int
    return struct.unpack('=i', stream.read(4))[0]


def Float(stream):
    # type: (Union[io.BinaryIO, io.BufferedReader]) -> float
    return struct.unpack('=f', stream.read(4))[0]


def String(stream):
    # type: (Union[io.BinaryIO, io.BufferedReader]) -> str
    length = Int(stream)
    if length <= 0:
        return ""
    byte = struct.unpack('=%ds' % length, stream.read(length))[0]
    s = byte
    if isinstance(s, bytes):
        s = s.decode()
    return s


def read_param(stream, *args):
    params = []
    for arg in args:
        params.append(arg(stream))
    return params