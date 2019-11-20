from ..node import Node

from typing import Union, List, Tuple
import numpy
import math

from .common import *


def _has_infered(node):
    # type: (Node) -> bool
    return node.has(Node.RetentionParam.shape) and node.has(Node.RetentionParam.dtype)


# pretty same as TS
def dynamic_padding_valid(input_size, static_padding, ksize, stride, dilation):
    # type: (Size2D, Padding2D, KSize2D, Stride2D, Dilation2D) -> Padding2D

    dynamic_padding = Padding2D()

    dynamic_padding.top = static_padding.top
    dynamic_padding.left = static_padding.left
    dynamic_padding.bottom = static_padding.bottom
    dynamic_padding.right = static_padding.right

    return dynamic_padding


class _out_pad(object):
    def __init__(self):
        self.out = 0    # output size
        self.pad_l = 0  # final padding (including static padding and dynamic padding)
        self.pad_r = 0  # final padding (including static padding and dynamic padding)


def forward_same(x, ksize, stride, dilation, padding):
    # type: (int, int, int, int, str) -> _out_pad
    out = _out_pad()

    idm = x
    dk = dilation * (ksize - 1) + 1
    odm = int((idm + stride - 1) / float(stride))
    padding_needed = max(0, (odm - 1) * stride + dk - idm)

    y = out.out
    pad_l = out.pad_l
    pad_r = out.pad_r

    def DEFINE_SAME_PADDING(A, B):
        A = padding_needed // 2
        B = padding_needed - A
        return A, B

    if padding != "SAME_UPPER":
        pad_l, pad_r = DEFINE_SAME_PADDING(pad_l, pad_r)
    else:
        pad_r, pad_l = DEFINE_SAME_PADDING(pad_r, pad_l)    # SAME_LOWER or SAME

    y = odm

    out.out = y
    out.pad_l = pad_l
    out.pad_r = pad_r

    return out


def conv2d_forward_same(x, padding, ksize, stride, dilation, padding_method):
    # type: (Size2D, Padding2D, KSize2D, Stride2D, Dilation2D, str) -> List[_out_pad]
    out = [_out_pad(), _out_pad()]
    out[0] = forward_same(x.height, ksize.height, stride.height, dilation.height, padding_method)
    out[1] = forward_same(x.width, ksize.width, stride.width, dilation.width, padding_method)
    return out


def dynamic_padding_same(input_size, static_padding, ksize, stride, dilation, padding_method):
    # type: (Size2D, Padding2D, KSize2D, Stride2D, Dilation2D, str) -> Padding2D

    dynamic_padding = Padding2D()

    dynamic_output_size = conv2d_forward_same(input_size, static_padding, ksize, stride, dilation, padding_method)

    expected_output_size = Size2D([dynamic_output_size[0].out, dynamic_output_size[1].out])
    expected_input_size = pooling2d_backward(expected_output_size, static_padding, ksize, stride)

    padding_height = (expected_input_size.height - input_size.height)
    padding_width = (expected_input_size.width - input_size.width)

    dynamic_padding.top = dynamic_output_size[0].pad_l
    dynamic_padding.left = dynamic_output_size[1].pad_l
    dynamic_padding.bottom = \
        static_padding.top + static_padding.bottom + (padding_height - dynamic_output_size[0].pad_l)
    dynamic_padding.right = \
        static_padding.left + static_padding.right + (padding_width - dynamic_output_size[1].pad_l)

    return dynamic_padding


def dragon_conv2d_padding(node):
    # type: (Node) -> Union[None, numpy.ndarray, List[List[int]]]
    x = node.inputs[0]
    w = node.inputs[1]

    assert isinstance(x, Node)
    assert isinstance(w, Node)

    if not _has_infered(x) or w.op != Node.Const:
        return None

    input_size = numpy.asarray(x.shape)
    stride = node.get("stride")
    dilation = node.get("dilation")
    static_padding = node.get("padding")
    static_padding = numpy.asarray(static_padding).reshape(4, 2)

    padding_method = str(node.get("padding_method"))
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
    ksize = KSize2D(w.shape[-2:])
    stride = Stride2D(stride[plant])
    dilation = Stride2D(dilation[plant])

    if padding_method == "NOTSET":
        padding_method = "VALID"
    if padding_method == "SAME":
        padding_method = "SAME_LOWER"

    dynamic_padding = Padding2D()

    if padding_method == "SAME_UPPER":
        dynamic_padding = dynamic_padding_same(input_size, static_padding, ksize, stride, dilation, "SAME_UPPER")
    elif padding_method == "SAME_LOWER":
        dynamic_padding = dynamic_padding_same(input_size, static_padding, ksize, stride, dilation, "SAME_LOWER")
    elif padding_method == "VALID":
        dynamic_padding = dynamic_padding_valid(input_size, static_padding, ksize, stride, dilation)
    else:
        return None

    return_padding = numpy.zeros([4, 2]).astype(numpy.int32)
    return_padding[plant[0], 0] = dynamic_padding.top
    return_padding[plant[0], 1] = dynamic_padding.bottom
    return_padding[plant[1], 0] = dynamic_padding.left
    return_padding[plant[1], 1] = dynamic_padding.right

    return return_padding
