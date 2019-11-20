from ..node import Node

from typing import Union, List, Tuple
import numpy
import math

from .common import *


def _has_infered(node):
    # type: (Node) -> bool
    return node.has(Node.RetentionParam.shape) and node.has(Node.RetentionParam.dtype)


def pooling2d_forward_notset(x, padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Size2D
    y = Size2D()
    y.height = int(math.floor((x.height + padding.top + padding.bottom - ksize.height) / float(stride.height) + 1))
    y.width = int(math.floor((x.width + padding.left + padding.right - ksize.width) / float(stride.width) + 1))
    return y


def dynamic_padding_notset(input_size, static_padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Padding2D

    dynamic_padding = Padding2D()

    dynamic_padding.top = static_padding.top
    dynamic_padding.left = static_padding.left
    dynamic_padding.bottom = static_padding.bottom \
        - (input_size.height + static_padding.top + static_padding.bottom - ksize.height) % stride.height
    dynamic_padding.right = static_padding.right \
        - (input_size.width + static_padding.left + static_padding.right - ksize.width) % stride.width


    return dynamic_padding


# VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
def forward_valid(input_spatial_shape, kernel_spatial_shape, strides_spatial_shape):
    # type(int, int, int) -> int
    return int(math.ceil((input_spatial_shape - kernel_spatial_shape + 1) / float(strides_spatial_shape)))


def pooling2d_forward_valid(x, padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Size2D
    y = Size2D()
    y.height = forward_valid(x.height + padding.top + padding.bottom, ksize.height, stride.height)
    y.width = forward_valid(x.width + padding.left + padding.right, ksize.width, stride.width)
    return y


def dynamic_padding_valid(input_size, static_padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Padding2D

    dynamic_padding = Padding2D()

    expected_output_size = pooling2d_forward_valid(input_size, static_padding, ksize, stride)
    expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride)
    dynamic_padding.top = static_padding.top
    dynamic_padding.left = static_padding.left
    dynamic_padding.bottom = static_padding.bottom + expected_input_size.height - input_size.height
    dynamic_padding.right = static_padding.right + expected_input_size.width - input_size.width

    return dynamic_padding


# SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
def forward_same(input_spatial_shape, kernel_spatial_shape, strides_spatial_shape):
    # type: (int, int, int) -> int
    return int(math.ceil(input_spatial_shape / float(strides_spatial_shape)))


def pooling2d_forward_same(x, padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Size2D
    y = Size2D()
    y.height = forward_same(x.height + padding.top + padding.bottom, ksize.height, stride.height)
    y.width = forward_same(x.width + padding.left + padding.right, ksize.width, stride.width)
    return y


def dynamic_padding_same_upper(input_size, static_padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Padding2D

    dynamic_padding = Padding2D()

    expected_output_size = pooling2d_forward_same(input_size, static_padding, ksize, stride)
    expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride)

    padding_height = (expected_input_size.height - input_size.height)
    padding_width = (expected_input_size.width - input_size.width)
    half_padding_height = padding_height // 2
    half_padding_width = padding_width // 2

    dynamic_padding.top = static_padding.top + half_padding_height
    dynamic_padding.left = static_padding.left + half_padding_height
    dynamic_padding.bottom = static_padding.bottom + (padding_height - half_padding_height)
    dynamic_padding.right = static_padding.right + (padding_width - half_padding_width)

    return dynamic_padding


def dynamic_padding_same_lower(input_size, static_padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Padding2D

    dynamic_padding = Padding2D()

    expected_output_size = pooling2d_forward_same(input_size, static_padding, ksize, stride)
    expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride)

    padding_height = (expected_input_size.height - input_size.height)
    padding_width = (expected_input_size.width - input_size.width)
    half_padding_height = padding_height // 2
    half_padding_width = padding_width // 2

    dynamic_padding.top = static_padding.top + (padding_height - half_padding_height)
    dynamic_padding.left = static_padding.left + (padding_width - half_padding_width)
    dynamic_padding.bottom = static_padding.bottom + half_padding_height
    dynamic_padding.right = static_padding.right + half_padding_width

    return dynamic_padding


def onnx_pooling2d_padding(node):
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

    auto_pad = str(node.get("auto_pad"))
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

    dynamic_padding = Padding2D()

    if auto_pad == "NOTSET":
        dynamic_padding = dynamic_padding_notset(input_size, static_padding, ksize, stride)
    elif auto_pad == "VALID":
        dynamic_padding = dynamic_padding_valid(input_size, static_padding, ksize, stride)
    elif auto_pad == "SAME_LOWER":
        dynamic_padding = dynamic_padding_same_lower(input_size, static_padding, ksize, stride)
    elif auto_pad == "SAME_UPPER":
        dynamic_padding = dynamic_padding_same_upper(input_size, static_padding, ksize, stride)
    else:
        return None

    return_padding = numpy.zeros([4, 2]).astype(numpy.int32)
    return_padding[plant[0], 0] = dynamic_padding.top
    return_padding[plant[0], 1] = dynamic_padding.bottom
    return_padding[plant[1], 0] = dynamic_padding.left
    return_padding[plant[1], 1] = dynamic_padding.right

    return return_padding
