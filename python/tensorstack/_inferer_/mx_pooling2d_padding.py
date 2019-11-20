from ..node import Node

from typing import Union, List, Tuple
import numpy
import math

from .common import *


def _has_infered(node):
    # type: (Node) -> bool
    return node.has(Node.RetentionParam.shape) and node.has(Node.RetentionParam.dtype)


def valid_pooling2d_forward(x, padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Size2D
    y = Size2D()
    y.height = int(math.floor((x.height + padding.top + padding.bottom - ksize.height) / float(stride.height) + 1))
    y.width = int(math.floor((x.width + padding.left + padding.right - ksize.width) / float(stride.width) + 1))
    return y


def dynamic_padding_valid(input_size, static_padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Padding2D

    dynamic_padding = Padding2D()

    dynamic_padding.top = static_padding.top
    dynamic_padding.left = static_padding.left
    dynamic_padding.bottom = static_padding.bottom \
        - (input_size.height + static_padding.top + static_padding.bottom - ksize.height) % stride.height
    dynamic_padding.right = static_padding.right \
        - (input_size.width + static_padding.left + static_padding.right - ksize.width) % stride.width

    return dynamic_padding


def mx_pooling2d_padding(node):
    # type: (Node) -> Union[None, numpy.ndarray, List[List[int]]]
    x = node.inputs[0]
    ksize = node.inputs[1]
    stride = node.inputs[2]

    assert isinstance(x, Node)
    assert isinstance(ksize, Node)
    assert isinstance(stride, Node)

    if not _has_infered(x) or ksize.op != Node.Const or stride.op != Node.Const:
        return None

    input_size = numpy.asarray(x.shape)
    ksize = numpy.asarray(ksize.get("value"))
    stride = numpy.asarray(stride.get("value"))
    static_padding = node.get("padding")
    static_padding = numpy.asarray(static_padding).reshape(4, 2)

    fmt = str(node.try_get("format", "NCHW"))
    plant = ()
    if fmt == "NCHW":
        plant = (2, 3)
    elif fmt == "NHWC":
        plant = (1, 2)
    else:
        return None
    plant = list(plant)

    input_size = Size2D(input_size[plant])
    static_padding = Padding2D(static_padding[plant])
    ksize = KSize2D(ksize[plant])
    stride = Stride2D(stride[plant])

    valid = bool(node.get("valid"))

    dynamic_padding = Padding2D()

    if valid:
        dynamic_padding = dynamic_padding_valid(input_size, static_padding, ksize, stride)
    else:
        dynamic_padding = static_padding

    return_padding = numpy.zeros([4, 2]).astype(numpy.int32)
    return_padding[plant[0], 0] = dynamic_padding.top
    return_padding[plant[0], 1] = dynamic_padding.bottom
    return_padding[plant[1], 0] = dynamic_padding.left
    return_padding[plant[1], 1] = dynamic_padding.right

    return return_padding
