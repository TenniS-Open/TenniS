from ..node import Node

from typing import Union, List, Tuple
import numpy
import math

from .common import *


def _has_infered(node):
    # type: (Node) -> bool
    return node.has(Node.RetentionParam.shape) and node.has(Node.RetentionParam.dtype)


# VALID: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
def forward_valid(x, ksize, pad_l, pad_r, stride, ceil_mode):
    # type: (int, int, int, int, int, bool) -> int
    y = 0
    if ceil_mode:
        y = int(math.ceil((x + pad_l + pad_r - ksize) / float(stride)) + 1)
    else:
        y = int(math.floor((x + pad_l + pad_r - ksize) / float(stride)) + 1)

    if (y - 1) * stride >= (x + pad_l + pad_r):
        y -= 1
    return y


def pooling2d_forward_valid(x, ksize, padding, stride, ceil_mode):
    # type: (Size2D, KSize2D, Padding2D, Stride2D, bool) -> Size2D
    y = Size2D()
    y.height = forward_valid(x.height, ksize.height, padding.top, padding.bottom, stride.height, ceil_mode)
    y.width = forward_valid(x.width, ksize.width, padding.left, padding.right, stride.width, ceil_mode)
    return y


def dynamic_padding_valid(input_size, static_padding, ksize, stride, ceil_mode):
    # type: (Size2D, Padding2D, KSize2D, Stride2D, bool) -> Padding2D
    dynamic_padding = Padding2D()

    expected_output_size = pooling2d_forward_valid(input_size, ksize, static_padding, stride, ceil_mode)
    expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride)
    dynamic_padding.top = static_padding.top
    dynamic_padding.left = static_padding.left
    dynamic_padding.bottom = static_padding.bottom + expected_input_size.height - input_size.height
    dynamic_padding.right = static_padding.right + expected_input_size.width - input_size.width

    return dynamic_padding


class _out_pad(object):
    def __init__(self):
        self.out = 0    # output size
        self.pad_l = 0  # final padding (including static padding and dynamic padding)
        self.pad_r = 0  # final padding (including static padding and dynamic padding)


# SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
def forward_same(x, ksize, stride, padding, ceil_mode):
    # type: (int, int, int, str, bool) -> _out_pad
    out = _out_pad()

    idm = x
    odm = int((idm + stride - 1) / float(stride))
    padding_needed = max(0, (odm - 1) * stride + ksize - idm)

    y = out.out
    pad_l = out.pad_l
    pad_r = out.pad_r

    def DEFINE_SAME_PADDING(A, B):
        A = padding_needed // 2
        B = padding_needed - A
        return A, B

    if padding == "SAME_UPPER":
        pad_l, pad_r = DEFINE_SAME_PADDING(pad_l, pad_r)
    else:
        pad_r, pad_l = DEFINE_SAME_PADDING(pad_r, pad_l)

    y = int(math.ceil(float(x) / stride))


    out.out = y
    out.pad_l = pad_l
    out.pad_r = pad_r

    return out


def pooling2d_forward_same(x, ksize, stride, padding, ceil_mode):
    # type: (Size2D, KSize2D, Stride2D, str, bool) -> List[_out_pad]
    out = [_out_pad(), _out_pad()]
    out[0] = forward_same(x.height, ksize.height, stride.height, padding, ceil_mode)
    out[1] = forward_same(x.width, ksize.width, stride.width, padding, ceil_mode)
    return out


def dynamic_padding_same(input_size, static_padding, ksize, stride, padding, ceil_mode):
    # type: (Size2D, Padding2D, KSize2D, Stride2D, str, bool) -> Padding2D

    dynamic_padding = Padding2D()

    dynamic_output_size = pooling2d_forward_same(input_size, ksize, stride, padding, ceil_mode)

    expected_output_size = Size2D([dynamic_output_size[0].out, dynamic_output_size[1].out])
    expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride)

    padding_height = (expected_input_size.height - input_size.height)
    padding_width = (expected_input_size.width - input_size.width)

    dynamic_padding.top = dynamic_output_size[0].pad_l
    dynamic_padding.left = dynamic_output_size[1].pad_l
    dynamic_padding.bottom = static_padding.top + static_padding.bottom + (padding_height - dynamic_output_size[0].pad_l)
    dynamic_padding.right = static_padding.left + static_padding.right + (padding_width - dynamic_output_size[1].pad_l)

    return dynamic_padding


def dragon_pooling2d_padding(node):
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

    ceil_mode = bool(node.try_get("ceil", True))

    if auto_pad == "NOTSET":
        auto_pad = "VALID"
    if auto_pad == "SAME":
        auto_pad = "SAME_LOWER"

    dynamic_padding = Padding2D()

    if auto_pad == "NOTSET":
        dynamic_padding =  dynamic_padding_valid(input_size, static_padding, ksize, stride, ceil_mode)
    elif auto_pad == "VALID":
        dynamic_padding = dynamic_padding_valid(input_size, static_padding, ksize, stride, ceil_mode)
    elif auto_pad == "SAME_LOWER":
        dynamic_padding = dynamic_padding_same(input_size, static_padding, ksize, stride, "SAME_LOWER", ceil_mode)
    elif auto_pad == "SAME_UPPER":
        dynamic_padding = dynamic_padding_same(input_size, static_padding, ksize, stride, "SAME_UPPER", ceil_mode)
    else:
        return None

    return_padding = numpy.zeros([4, 2]).astype(numpy.int32)
    return_padding[plant[0], 0] = dynamic_padding.top
    return_padding[plant[0], 1] = dynamic_padding.bottom
    return_padding[plant[1], 0] = dynamic_padding.left
    return_padding[plant[1], 1] = dynamic_padding.right

    return return_padding
