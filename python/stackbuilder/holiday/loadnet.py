#!python
# coding: UTF-8
"""
author: kier
"""

from .proto import HolidayCNN_proto_pb2 as hd

import io
import struct
import sys


def compatible_string(obj):
    # type: (object) -> object
    if sys.version > '3':
        pass
    else:
        if isinstance(obj, unicode):
            return str(obj)
    return obj


def read_long(fi):
    # type: (Union[io.BinaryIO, io.BufferedReader]) -> int
    return struct.unpack('=q', fi.read(8))[0]


def write_long(fo, i):
    # type: (Union[io.BinaryIO, io.BufferedWriter], int) -> None
    byte = struct.pack('=q', i)
    fo.write(byte)


def read_int(fi):
    # type: (Union[io.BinaryIO, io.BufferedReader]) -> int
    return struct.unpack('=i', fi.read(4))[0]


def write_int(fo, i):
    # type: (Union[io.BinaryIO, io.BufferedWriter], int) -> None
    byte = struct.pack('=i', i)
    fo.write(byte)


def read_long_string(fi):
    # type: (Union[io.BinaryIO, io.BufferedReader]) -> str
    length = read_long(fi)
    byte = struct.unpack('=%ds' % length, fi.read(length))[0]
    s = byte
    if isinstance(s, bytes):
        s = s.decode()
    return s


def write_long_string(fo, s):
    # type: (Union[io.BinaryIO, io.BufferedWriter], Union[str, bytes]) -> None
    s = compatible_string(s)
    if isinstance(s, str):
        s = s.encode()
    elif isinstance(s, bytes):
        pass
    else:
        raise Exception("Can not write type={} as string".format(type(s)))
    byte = struct.pack('=q%ds' % len(s), len(s), s)
    fo.write(byte)


def read_short_string(fi):
    # type: (Union[io.BinaryIO, io.BufferedReader]) -> str
    length = read_int(fi)
    byte = struct.unpack('=%ds' % length, fi.read(length))[0]
    s = byte
    if isinstance(s, bytes):
        s = s.decode()
    return s


def write_short_string(fo, s):
    # type: (Union[io.BinaryIO, io.BufferedWriter], Union[str, bytes]) -> None
    s = compatible_string(s)
    if isinstance(s, str):
        s = s.encode()
    elif isinstance(s, bytes):
        pass
    else:
        raise Exception("Can not write type={} as string".format(type(s)))
    byte = struct.pack('=i%ds' % len(s), len(s), s)
    fo.write(byte)


def read_layer(fi):
    # type: (Union[io.BinaryIO, io.BufferedReader]) -> hd.Holiday_LayerParameter
    length = read_long(fi)
    buffer = struct.unpack('=%ds' % length, fi.read(length))[0]
    # buffer = fi.read(length)
    layer = hd.Holiday_LayerParameter()
    layer.ParseFromString(buffer)
    return layer


def write_layer(fo, layer):
    # type: (Union[io.BinaryIO, io.BufferedWriter], hd.Holiday_LayerParameter) -> None
    s = layer.SerializeToString()
    byte = struct.pack('=q%ds' % len(s), len(s), s)
    fo.write(byte)


class Header:
    def __init__(self):
        self.__feature_size = 0
        self.__channels = 0
        self.__width = 0
        self.__height = 0
        self.__blob_name = ''

    @property
    def feature_size(self):
        return self.__feature_size

    @feature_size.setter
    def feature_size(self, value):
        self.__feature_size = value

    @property
    def channels(self):
        return self.__channels

    @channels.setter
    def channels(self, value):
        self.__channels = value

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, value):
        self.__width = value

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, value):
        self.__height = value

    @property
    def blob_name(self):
        return self.__blob_name

    @blob_name.setter
    def blob_name(self, value):
        self.__blob_name = value

    def load(self, fi):
        # type: (Union[io.BinaryIO, io.BufferedReader]) -> Header
        self.__feature_size = read_int(fi)
        self.__channels = read_int(fi)
        self.__width = read_int(fi)
        self.__height = read_int(fi)
        self.__blob_name = read_short_string(fi)
        return self

    def save(self, fo):
        # type: (Union[io.BinaryIO, io.BufferedWriter]) -> None
        write_int(fo, self.__feature_size)
        write_int(fo, self.__channels)
        write_int(fo, self.__width)
        write_int(fo, self.__height)
        write_short_string(fo, self.__blob_name)

    def __str__(self):
        return "[{}, {}, {}, {}, {}]".format(
            self.__feature_size, self.__channels, self.__width, self.__height, self.__blob_name)

    def __repr__(self):
        return "[{}, {}, {}, {}, {}]".format(
            self.__feature_size, self.__channels, self.__width, self.__height, self.__blob_name)


class OPType(object):
    Enum_ConvolutionLayer = 0
    Enum_EltwiseLayer = 1
    Enum_ConcatLayer = 2
    Enum_ExpLayer = 3
    Enum_InnerProductLayer = 4
    Enum_LRNLayer = 5
    Enum_MemoryDataLayer = 6
    Enum_PoolingLayer = 7
    Enum_PowerLayer = 8
    Enum_ReLULayer = 9
    Enum_SoftmaxLayer = 10
    Enum_SliceLayer = 11
    Enum_BatchNormliseLayer = 13
    Enum_ScaleLayer = 14
    Enum_SplitLayer = 15
    Enum_PreReLULayer = 16
    Enum_DeconvolutionLayer = 17
    Enum_CropLayer = 18
    Enum_SigmoidLayer = 19
    Enum_FinallyLayer = 1001

    # for tf
    Enum_SpaceToBatchNDLayer = 20
    Enum_BatchToSpaceNDLayer = 21

    # for tf
    Enum_ReshapeLayer = 22
    Enum_RealMulLayer = 23

    Enum_ShapeIndexPatchLayer = 31

    EnumString = {
        Enum_ConvolutionLayer: "Convolution",
        Enum_EltwiseLayer: "Eltwise",
        Enum_ConcatLayer: "Concat",
        Enum_ExpLayer: "Exp",
        Enum_InnerProductLayer: "InnerProduct",
        Enum_LRNLayer: "LRN",
        Enum_MemoryDataLayer: "MemoryData",
        Enum_PoolingLayer: "Pooling",
        Enum_PowerLayer: "Power",
        Enum_ReLULayer: "ReLU",
        Enum_SoftmaxLayer: "Softmax",
        Enum_SliceLayer: "Slice",
        Enum_BatchNormliseLayer: "BatchNormlise",
        Enum_ScaleLayer: "Scale",
        Enum_SplitLayer: "Split",
        Enum_PreReLULayer: "PreReLU",
        Enum_DeconvolutionLayer: "Deconvolution",
        Enum_CropLayer: "Crop",
        Enum_SigmoidLayer: "Sigmoid",
        Enum_FinallyLayer: "Finally",

        # for tf
        Enum_SpaceToBatchNDLayer: "SpaceToBatchND",
        Enum_BatchToSpaceNDLayer: "BatchToSpaceND",

        # for tf
        Enum_ReshapeLayer: "Reshape",
        Enum_RealMulLayer: "RealMul",

        Enum_ShapeIndexPatchLayer: "ShapeIndexPatch",
    }


class Net(object):
    def __init__(self):
        self.__blob_names = []
        self.__layer_names = []
        self.__layers = []
        self.__last_layer = None

    @property
    def blob_names(self):
        # type: () -> List[str]
        return self.__blob_names

    @blob_names.setter
    def blob_names(self, value):
        # type: (List[str]) -> None
        self.__blob_names = value

    @property
    def layer_names(self):
        # type: () -> List[str]
        return self.__layer_names

    @layer_names.setter
    def layer_names(self, value):
        # type: (List[str]) -> None
        self.__layer_names = value

    @property
    def layers(self):
        # type: () -> List[hd.Holiday_LayerParameter]
        return self.__layers

    @layers.setter
    def layers(self, value):
        # type: (List[hd.Holiday_LayerParameter]) -> None
        self.__layers = value

    @property
    def last_layer(self):
        return self.__last_layer

    @last_layer.setter
    def last_layer(self, value):
        self.__last_layer = value

    def load(self, fi):
        # type: (Union[io.BinaryIO, io.BufferedReader]) -> Net
        self.__blob_names = []
        self.__layer_names = []
        self.__layers = []
        self.__last_layer = None

        blob_names_length = read_long(fi)
        for i in range(blob_names_length):
            self.__blob_names.append(read_long_string(fi))

        layer_names_length = read_long(fi)
        for i in range(layer_names_length):
            self.__layer_names.append(read_long_string(fi))

        # must has last layer
        while True:
            layer = read_layer(fi)
            # print("{}: {}".format(layer.layer_index, layer.name))
            if layer.type == OPType.Enum_FinallyLayer:
                self.__last_layer = layer
                break
            self.__layers.append(layer)

        # for last
        assert self.__last_layer.type == OPType.Enum_FinallyLayer

        return self

    def save(self, fo, header=None):
        # type: (Union[io.BinaryIO, io.BufferedWriter], Header) -> None
        if header is not None:
            header.save(fo)

        write_long(fo, len(self.__blob_names))
        for blob_name in self.__blob_names:
            write_long_string(fo, blob_name)

        write_long(fo, len(self.__layer_names))
        for layer_name in self.__layer_names:
            write_long_string(fo, layer_name)

        for layer in self.__layers:
            write_layer(fo, layer)

        write_layer(fo, self.__last_layer)


def load_net(path, has_header=True):
    # type: (Union[str, file], bool) -> (Header, Net)
    path = compatible_string(path)
    if isinstance(path, str):
        with open(path, 'rb') as fi:
            return load_net_in_memory(fi, has_header=has_header)
    return load_net_in_memory(path, has_header=has_header)


def load_net_in_memory(stream, has_header=True):
    # type: (Any, bool) -> (Header, Net)
    header = None
    if has_header:
        header = Header().load(stream)
    net = Net().load(stream)
    return header, net


def save_net(path, net, header=None):
    # type: (str, Net, Header) -> None
    with open(path, 'wb') as fo:
        net.save(fo, header)


if __name__ == '__main__':
    header, net = load_net("/Users/seetadev/Documents/Files/models/model/VIPLPoseEstimation1.1.0.ext.dat")

    print(header)
    print(net.blob_names)
    print(net.layer_names)
    print(len(net.layers))
    print(net.layers[-1])

    save_net("test.ext.dat", net, header)
    header, net = load_net("test.ext.dat")
    save_net("test.ext.dat", net, header)
    header2, net2 = load_net("test.ext.dat")

    print(header2)
    print(net2.blob_names)
    print(net2.layer_names)
    print(len(net2.layers))
    print(net.layers[-1])

    print()
