#!/usr/bin/env python

"""
:author Kier
"""

from .node import Node
from . import menu as menu
from . import device as device
from . import tensor

import numpy

import sys
if sys.version > '3':
    long = int


class Name(object):
    NCHW = "NCHW"
    NHWC = "NHWC"

    class Layer(object):
        dimshuffle = "_dimshuffle"
        transpose = "_transpose"
        reshape = "_reshape"
        conv2d = "conv2d"
        conv2d_v2 = "conv2d_v2"
        # conv2d_bias = "conv2d_bias"
        # padding_conv2d_bias = "padding_conv2d_bias"
        shape = "_shape"
        pad = "pad"
        depthwise_conv2d = "depthwise_conv2d"
        depthwise_conv2d_v2 = "depthwise_conv2d_v2"
        # depthwise_conv2d_bias = "depthwise_conv2d_bias"
        # padding_depthwise_conv2d_bias = "padding_depthwise_conv2d_bias"
        add_bias = "add_bias"
        batch_norm = "batch_norm"
        batch_scale = "batch_scale"
        fused_batch_norm = "fused_batch_norm"
        add = "add"
        sub = "sub"
        mul = "mul"
        div = "div"
        inner_prod = "inner_prod"
        relu = "relu"
        prelu = "prelu"
        relu_max = "relu_max"
        sigmoid = "sigmoid"
        softmax = "softmax"
        concat = "concat"
        flatten = "flatten"
        to_float = "to_float"
        pooling2d = "pooling2d"
        pooling2d_v2 = "pooling2d_v2"
        resize2d = "_resize2d"
        copy = "_copy"
        prewhiten = "prewhiten"
        cast = "_cast"
        reshape_v2 = "_reshape_v2"
        global_pooling2d = "global_pooling2d"
        limit = "_limit"
        crop_nd = "crop_nd"
        chunk = "chunk"
        squeeze = "squeeze"
        rsqrt = "rsqrt"
        sample2d = "sample2d"

    dim = "dim"
    shuffle = "shuffle"
    value = "value"
    permute = "permute"
    shape = "shape"
    format = "format"
    padding = "padding"
    padding_value = "padding_value"
    stride = "stride"
    dilation = "dilation"
    epsilon = "epsilon"
    max = "max"
    slope = "slope"
    type = "type"
    padding_type = "padding_type"
    ksize = "ksize"
    device = "device"
    smooth = "smooth"
    dtype = "dtype"
    shift = "shfit"
    chunks = "chunks"
    axes = "axes"
    transpose = "transpose"
    scale = "scale"


class Default(object):
    @staticmethod
    def padding():
        return [[0, 0], [0, 0], [0, 0], [0, 0]]

    @staticmethod
    def ksize():
        return [1, 1, 1, 1]

    @staticmethod
    def stride():
        return [1, 1, 1, 1]

    @staticmethod
    def dilation():
        return [1, 1, 1, 1]

    @staticmethod
    def padding_value():
        return 0


class Type(object):
    class resize2d_type(object):
        linear = 0
        cubic = 1
        nearest = 2

    class padding_type(object):
        black = 0
        copy = 1
        loop = 2
        white = 3   # denominator force be ksize * ksize

    class pooling_type(object):
        max = 0
        avg = 1


def to_const(value, name=None):
    # type: (Any, str) -> numpy.ndarray
    if isinstance(value, Node):
        if value.op == Node.Const:
            value = value.get(Name.value)
        else:
            raise Exception("Param %s not support dynamic Node".format(name))
    return value


def to_node(value, name=None, device=None, dtype=None):
    if isinstance(value, Node):
        return value
    if dtype is not None:
        value = tensor.from_any(value, dtype=dtype)
    return menu.data(name=name, value=value, device=device)


def dimsuffle(name, x, dim, shuffle):
    assert isinstance(x, Node)

    dim = to_const(dim, "dim")
    shuffle = to_const(shuffle, "shuffle")

    node = menu.op(name=name, op_name=Name.Layer.dimshuffle, inputs=[x, ])
    node.set(Name.dim, dim, numpy.int32)
    node.set(Name.shuffle, shuffle, numpy.int32)

    return node


def transpose(name, x, pemute=None):
    assert isinstance(x, Node)

    node = menu.op(name=name, op_name=Name.Layer.transpose, inputs=[x, ])
    if pemute is not None:
        pemute = to_const(pemute, "pemute")
        node.set(Name.permute, pemute, numpy.int32)

    return node


def reshape_v2(name, x, shape):
    assert isinstance(x, Node)

    shape = to_node(shape, "shape", dtype=numpy.int32, device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.reshape_v2, inputs=[x, shape])

    return node


def reshape(name, x, shape):
    assert isinstance(x, Node)

    node = None
    if isinstance(shape, Node):
        node = menu.op(name=name, op_name=Name.Layer.reshape_v2, inputs=[x, shape])
    else:
        node = menu.op(name=name, op_name=Name.Layer.reshape, inputs=[x,])
        node.set(Name.shape, shape, numpy.int32)

    return node


def rgb2bgr(name, x, format=Name.NCHW):
    assert isinstance(x, Node)
    assert format == Name.NCHW or format == Name.NHWC
    if format == Name.NCHW:
        return dimsuffle(name=name, x=x, dim=1, shuffle=[2, 1, 0])
    else:
        return dimsuffle(name=name, x=x, dim=3, shuffle=[2, 1, 0])


bgr2rgb = rgb2bgr


def NCHW2NHWC(name, x):
    assert isinstance(x, Node)

    return transpose(name=name, x=x, pemute=[0, 2, 3, 1])


def NHWC2NCHW(name, x):
    assert isinstance(x, Node)

    return transpose(name=name, x=x, pemute=[0, 3, 1, 2])

def format4h(format):
    if format == Name.NCHW:
        return 2
    elif format == Name.NHWC:
        return 1
    else:
        raise RuntimeError("format: {}".format(format))


def adjust4d(format, base, shape):
    h = format4h(format)
    if shape is None:
        return base
    if isinstance(shape, (int, long)):
        base[h] = shape
        base[h + 1] = shape
    elif isinstance(shape, (tuple, list)) or isinstance(shape, numpy.ndarray):
        numpy_shape = numpy.asarray(shape, dtype=numpy.int32)
        if numpy_shape.shape == (1,):
            base[h] = numpy_shape[0]
            base[h + 1] = numpy_shape[0]
        elif numpy_shape.shape == (2,):
            base[h] = numpy_shape[0]
            base[h + 1] = numpy_shape[1]
        elif numpy_shape.shape == (4,):
            base = shape
        else:
            raise RuntimeError("{}".format(shape))
    else:
        return RuntimeError("type={}".format(type(shape)))

    return base


def adjust4x2d(format, base, shape):
    h = format4h(format)
    if shape is None:
        return base
    if isinstance(shape, (int, long)):
        base[h][0] = shape
        base[h][1] = shape
        base[h + 1][0] = shape
        base[h + 1][1] = shape
    elif isinstance(shape, (tuple, list)) or isinstance(shape, numpy.ndarray):
        numpy_shape = numpy.asarray(shape, dtype=numpy.int32)
        if numpy_shape.shape == (2,):
            base[h][0] = numpy_shape[0]
            base[h][1] = numpy_shape[0]
            base[h + 1][0] = numpy_shape[1]
            base[h + 1][1] = numpy_shape[1]
        if numpy_shape.shape == (4,):
            base[0][0] = numpy_shape[0]
            base[0][1] = numpy_shape[0]
            base[1][0] = numpy_shape[1]
            base[1][1] = numpy_shape[1]
            base[2][0] = numpy_shape[2]
            base[2][1] = numpy_shape[2]
            base[3][0] = numpy_shape[3]
            base[3][1] = numpy_shape[3]
        elif numpy_shape.shape == (2, 2):
            base[h][0] = numpy_shape[0, 0]
            base[h][1] = numpy_shape[0, 1]
            base[h + 1][0] = numpy_shape[1, 0]
            base[h + 1][1] = numpy_shape[1, 1]
        elif numpy_shape.shape == (4, 2):
            base = shape
        else:
            raise RuntimeError("{}".format(shape))
    else:
        return RuntimeError("type={}".format(type(shape)))

    return base


def adjust_padding(padding, format=Name.NCHW):
    if padding is None:
        return Default.padding()
    if isinstance(padding, Node):
        return padding
    try:
        return adjust4x2d(format, Default.padding(), padding)
    except RuntimeError as e:
        raise RuntimeError("Not support padding: {}".format(e))


def adjust_stride(stride, format=Name.NCHW):
    if stride is None:
        return Default.stride()
    if isinstance(stride, Node):
        return stride
    try:
        return adjust4d(format, Default.stride(), stride)
    except RuntimeError as e:
        raise RuntimeError("Not support stride: {}".format(e))


def adjust_dilation(dilation, format=Name.NCHW):
    if dilation is None:
        return Default.dilation()
    if isinstance(dilation, Node):
        return dilation
    try:
        return adjust4d(format, Default.dilation(), dilation)
    except RuntimeError as e:
        raise RuntimeError("Not support dilation: {}".format(e))


def adjust_ksize(ksize, format=Name.NCHW):
    if ksize is None:
        return Default.ksize()
    if isinstance(ksize, Node):
        return ksize
    try:
        return adjust4d(format, Default.ksize(), ksize)
    except RuntimeError as e:
        raise RuntimeError("Not support ksize: {}".format(e))


def conv2d(name, x, w,
           format=Name.NCHW,
           padding=None,
           padding_value=None,
           stride=None,
           dilation=None):
    assert isinstance(x, Node)

    padding = adjust_padding(padding, format=format)
    stride = adjust_stride(stride, format=format)
    dilation = adjust_dilation(dilation, format=format)

    if padding is None:
        padding = Default.padding()
    if padding_value is None:
        padding_value = Default.padding_value()
    if stride is None:
        stride = Default.stride()
    if dilation is None:
        dilation = Default.dilation()
    w = to_node(w, name="_const_" + name + "_weights")

    node = None

    if isinstance(padding, Node):
        node = menu.op(name=name, op_name=Name.Layer.conv2d_v2, inputs=[x, padding, w])
        # node.set(Name.padding, Default.padding())
    else:
        node = menu.op(name=name, op_name=Name.Layer.conv2d, inputs=[x, w])
        node.set(Name.padding, padding, numpy.int32)

    node.set(Name.format, format)
    node.set(Name.padding_value, padding_value)
    node.set(Name.stride, stride, numpy.int32)
    node.set(Name.dilation, dilation, numpy.int32)

    return node


def shape(name, x):
    assert isinstance(x, Node)

    return menu.op(name=name, op_name=Name.Layer.shape, inputs=[x, ])


def pad(name, x, padding, padding_value=None):
    assert isinstance(x, Node)

    if padding_value is None:
        padding_value = Default.padding_value()
    padding = to_node(padding, name="_const_" + name + "_padding", dtype=numpy.int32, device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.pad, inputs=[x, padding])
    node.set(Name.padding_value, padding_value)

    return node


def depthwise_conv2d(name, x, w,
                     format=Name.NCHW,
                     padding=None,
                     padding_value=None,
                     stride=None,
                     dilation=None):
    assert isinstance(x, Node)

    padding = adjust_padding(padding, format=format)
    stride = adjust_stride(stride, format=format)
    dilation = adjust_dilation(dilation, format=format)

    if padding is None:
        padding = Default.padding()
    if padding_value is None:
        padding_value = Default.padding_value()
    if stride is None:
        stride = Default.stride()
    if dilation is None:
        dilation = Default.dilation()
    w = to_node(w, name="_const_" + name + "_weights")

    node = None
    if isinstance(padding, Node):
        node = menu.op(name=name, op_name=Name.Layer.depthwise_conv2d_v2, inputs=[x, padding, w])
        # node.set(Name.padding, Default.padding())
    else:
        node = menu.op(name=name, op_name=Name.Layer.depthwise_conv2d, inputs=[x, w])
        node.set(Name.padding, padding, numpy.int32)

    node.set(Name.format, format)
    node.set(Name.padding_value, padding_value)
    node.set(Name.stride, stride, numpy.int32)
    node.set(Name.dilation, dilation, numpy.int32)

    return node


def add_bias(name, x, b, dim=1, format=None):
    assert isinstance(x, Node)
    assert format is None or format == Name.NCHW or format == Name.NHWC

    b = to_node(b, name="_const_" + name + "_bias")

    node = menu.op(name=name, op_name=Name.Layer.add_bias, inputs=[x, b])

    # dim = 1
    if format is not None:
        if format == Name.NCHW:
            dim = 1
        else:
            dim = 3

    if format is not None:
        node.set(Name.format, format)
    node.set(Name.dim, dim, numpy.int32)

    return node


def batch_norm(name, x, mean, variance, dim, epsilon):
    assert isinstance(x, Node)
    mean = to_node(mean, name="_const_" + name + "_mean")
    variance = to_node(variance, name="_const_" + name + "_variance")

    node = menu.op(name=name, op_name=Name.Layer.batch_norm, inputs=[x, mean, variance])
    node.set(Name.dim, dim, numpy.int32)
    node.set(Name.epsilon, epsilon)

    return node


def batch_scale(name, x, scale, bias, dim):
    assert isinstance(x, Node)
    scale = to_node(scale, name="_const_" + name + "_mean")
    bias = to_node(bias, name="_const_" + name + "_variance")

    node = menu.op(name=name, op_name=Name.Layer.batch_scale, inputs=[x, scale, bias])
    node.set(Name.dim, dim, numpy.int32)

    return node


def fused_batch_norm(name, x, mean, variance, scale, bias, dim, epsilon):
    assert isinstance(x, Node)
    mean = to_node(mean, name="_const_" + name + "_mean")
    variance = to_node(variance, name="_const_" + name + "_variance")
    scale = to_node(scale, name="_const_" + name + "_mean")
    bias = to_node(bias, name="_const_" + name + "_variance")

    node = menu.op(name=name, op_name=Name.Layer.fused_batch_norm, inputs=[x, mean, variance, scale, bias])
    node.set(Name.dim, dim, numpy.int32)
    node.set(Name.epsilon, epsilon)

    return node


def add(name, lhs, rhs):
    lhs = to_node(lhs, name="_const_" + name + "_lhs")
    rhs = to_node(rhs, name="_const_" + name + "_rhs")

    node = menu.op(name=name, op_name=Name.Layer.add, inputs=[lhs, rhs])

    return node


def sub(name, lhs, rhs):
    lhs = to_node(lhs, name="_const_" + name + "_lhs")
    rhs = to_node(rhs, name="_const_" + name + "_rhs")

    node = menu.op(name=name, op_name=Name.Layer.sub, inputs=[lhs, rhs])

    return node


def mul(name, lhs, rhs):
    lhs = to_node(lhs, name="_const_" + name + "_lhs")
    rhs = to_node(rhs, name="_const_" + name + "_rhs")

    node = menu.op(name=name, op_name=Name.Layer.mul, inputs=[lhs, rhs])

    return node


def div(name, lhs, rhs):
    lhs = to_node(lhs, name="_const_" + name + "_lhs")
    rhs = to_node(rhs, name="_const_" + name + "_rhs")

    node = menu.op(name=name, op_name=Name.Layer.div, inputs=[lhs, rhs])

    return node


def inner_prod(name, lhs, rhs, transpose=False):
    lhs = to_node(lhs, name="_const_" + name + "_lhs")
    rhs = to_node(rhs, name="_const_" + name + "_rhs")

    node = menu.op(name=name, op_name=Name.Layer.inner_prod, inputs=[lhs, rhs])

    if transpose:
        node.set(Name.transpose, transpose)

    return node


def relu(name, x):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.relu, inputs=[x, ])
    return node


def relu_max(name, x, max):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.relu_max, inputs=[x, ])
    node.set(Name.max, max)
    return node


def prelu(name, x, dim, slope):
    assert isinstance(x, Node)
    slope = to_node(value=slope, name=name + "_slope")
    node = menu.op(name=name, op_name=Name.Layer.prelu, inputs=[x, slope])
    node.set(Name.dim, dim, numpy.int32)
    return node


def sigmoid(name, x):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.sigmoid, inputs=[x, ])
    return node


def softmax(name, x, dim, smooth=True):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.softmax, inputs=[x, ])
    node.set(Name.dim, dim, numpy.int32)
    node.set(Name.smooth, smooth, numpy.bool)
    return node


def concat(name, inputs, dim):
    for x in inputs:
        assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.concat, inputs=inputs)
    node.set(Name.dim, dim, numpy.int32)
    return node


def flatten(name, x, dim=1):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.flatten, inputs=[x, ])
    node.set(Name.dim, dim, numpy.int32)
    return node


def to_float(name, x):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.to_float, inputs=[x, ])
    return node


def resize2d(name, x, size, type=Type.resize2d_type.linear):
    assert isinstance(x, Node)

    size = to_node(size, name="_const_" + name + "_size", dtype=numpy.int32, device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.resize2d, inputs=[x, size])
    node.set(Name.type, type, numpy.int32)

    return node


def pooling2d_v2(name, x, ksize, stride, type=Type.pooling_type.max, format=Name.NCHW,
              padding=None,
              padding_type=Type.padding_type.black):
    assert isinstance(x, Node)

    padding = adjust_padding(padding, format=format)
    stride = adjust_stride(stride, format=format)
    ksize = adjust_ksize(ksize, format=format)

    if padding is None:
        padding = Default.padding()

    padding = to_node(padding, name="_const_" + name + "_padding", dtype=numpy.int32, device=device.CPU)
    ksize = to_node(ksize, name="_const_" + name + "_ksize", dtype=numpy.int32, device=device.CPU)
    stride = to_node(stride, name="_const_" + name + "_stride", dtype=numpy.int32, device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.pooling2d_v2, inputs=[x, padding, ksize, stride])
    node.set(Name.format, format)
    node.set(Name.type, type, numpy.int32)
    node.set(Name.padding_type, padding_type, numpy.int32)

    return node


def pooling2d(name, x, ksize, stride, type=Type.pooling_type.max, format=Name.NCHW,
              padding=None,
              padding_type=Type.padding_type.black):
    assert isinstance(x, Node)

    padding = adjust_padding(padding, format=format)
    stride = adjust_stride(stride, format=format)
    ksize = adjust_ksize(ksize, format=format)

    if padding is None:
        padding = Default.padding()

    if isinstance(ksize, Node) or isinstance(stride, Node) or isinstance(padding, Node):
        return pooling2d_v2(name=name, x=x,
                            ksize=ksize, stride=stride,
                            type=type, format=format, padding=padding, padding_type=padding_type)

    padding = to_const(padding, name="padding")
    ksize = to_const(ksize, name="ksize")
    stride = to_const(stride, name="stride")

    node = menu.op(name=name, op_name=Name.Layer.pooling2d, inputs=[x])
    node.set(Name.padding, padding, numpy.int32)
    node.set(Name.ksize, ksize, numpy.int32)
    node.set(Name.stride, stride, numpy.int32)
    node.set(Name.format, format)
    node.set(Name.type, type, numpy.int32)
    node.set(Name.padding_type, padding_type, numpy.int32)

    return node


def copy(name, x):
    assert isinstance(x, Node) or isinstance(x, (tuple, list))
    node = None
    if isinstance(x, Node):
        node = menu.op(name=name, op_name=Name.Layer.copy, inputs=[x, ])
    elif isinstance(x, (tuple, list)):
        for input in x:
            assert isinstance(input, Node)
        node = menu.op(name=name, op_name=Name.Layer.copy, inputs=x, output_count=len(x))
    else:
        raise NotImplementedError("type(x) = {}".format(type(x)))
    return node


def prewhiten(name, x):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.prewhiten, inputs=[x, ])
    return node


def cast(name, x, dtype):
    assert isinstance(x, Node)

    dtype = to_const(dtype, "dtype")

    node = menu.op(name=name, op_name=Name.Layer.cast, inputs=[x, ])
    node.set(Name.dtype, dtype, numpy.int32)

    return node


def global_pooling2d(name, x, type=Type.pooling_type.max, format=Name.NCHW):
    assert isinstance(x, Node)

    node = menu.op(name=name, op_name=Name.Layer.global_pooling2d, inputs=[x])
    node.set(Name.format, format)
    node.set(Name.type, type, numpy.int32)

    return node


def limit(name, x, shape):
    assert isinstance(x, Node)

    shape = to_const(shape, "shape")
    node = menu.op(name=name, op_name=Name.Layer.limit, inputs=[x, ])
    node.set(Name.shape, shape, numpy.int32)

    return node


def crop_nd(name, x, size, shift=None):
    assert isinstance(x, Node)

    size = to_node(size, "size", dtype=numpy.int32, device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.crop_nd, inputs=[x, size])
    if shift is not None:
        shift = to_const(shift, "shift")
        node.set(Name.shift, shift, numpy.int32)

    return node


def chunk(name, x, chunks, dim=0):
    assert isinstance(x, Node)

    chunks = to_const(chunks, "chunks")
    dim = to_const(dim, "dim")

    node = menu.op(name=name, op_name=Name.Layer.chunk, inputs=[x,])
    node.set(Name.chunks, chunks, numpy.int32)
    node.set(Name.dim, dim, numpy.int32)

    outputs = [menu.field(name=name + ":" + str(i), input=node, offset=i) for i in range(int(chunks))]

    return outputs


def squeeze(name, x, axes=None):
    assert isinstance(x, Node)

    # operator
    node = menu.op(name=name, op_name=Name.Layer.squeeze, inputs=[x, ])
    if axes is not None:
        axes = to_const(axes, "axes")
        node.set(Name.axes, axes, numpy.int32)

    return node


def rsqrt(name, x):
    assert isinstance(x, Node)

    # operator
    node = menu.op(name=name, op_name=Name.Layer.rsqrt, inputs=[x, ])

    return node


def sample2d(name, x, scale, type=Type.resize2d_type.nearest, dim=-2):
    assert isinstance(x, Node)

    node = menu.op(name=name, op_name=Name.Layer.sample2d, inputs=[x])
    node.set(Name.type, type, numpy.int32)
    node.set(Name.scale, scale, numpy.float32)
    node.set(Name.dim, dim, numpy.int32)

    return node

