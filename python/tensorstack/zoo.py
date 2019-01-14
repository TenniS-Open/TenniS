#!/usr/bin/env python

"""
:author Kier
"""

from node import Node
import menu as menu
import device as device


class Name(object):
    NCHW = "NCHW"
    NHWC = "NHWC"

    class Layer(object):
        dimshuffle = "_dimshuffle"
        transpose = "_transpose"
        reshape = "_reshape"
        conv2d = "conv2d"
        padding_conv2d = "padding_conv2d"
        # conv2d_bias = "conv2d_bias"
        # padding_conv2d_bias = "padding_conv2d_bias"
        shape = "_shape"
        pad = "pad"
        depthwise_conv2d = "depthwise_conv2d"
        padding_depthwise_conv2d = "padding_depthwise_conv2d"
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
        resize2d = "resize2d"

    dim = "dim"
    shuffle = "shuffle"
    value = "value"
    permute = "permute"
    shape = "shape"
    format = "format"
    padding = "padding"
    padding_value = "padding_value"
    stride = "stride"
    dialations = "dialations"
    epsilon = "epsilon"
    max = "max"
    slope = "slope"
    type = "type"
    padding_type = "padding_type"
    ksize = "ksize"


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
    def dialations():
        return [1, 1, 1, 1]

    @staticmethod
    def padding_value():
        return 0


class Type(object):
    class resize2d_type(object):
        linear = 0
        cubic = 1

    class padding_type(object):
        black = 0
        copy = 1
        loop = 2

    class pooling_type(object):
        max = 0
        avg = 1


def to_const(value, name=None):
    if isinstance(value, Node):
        if value.op == Node.Const:
            value = value.get(Name.value)
        else:
            raise Exception("Param %s not support dynamic Node".format(name))
    return value


def to_node(value, name=None, device=None):
    if isinstance(value, Node):
        return value
    return menu.data(name=name, value=value, device=device)


def dimsuffle(name, x, dim, shuffle):
    assert isinstance(x, Node)

    dim = to_const(dim, "dim")
    shuffle = to_const(shuffle, "shuffle")

    node = menu.op(name=name, op_name=Name.Layer.dimshuffle, inputs=[x, ])
    node.set(Name.dim, dim)
    node.set(Name.shuffle, shuffle)

    return node


def transpose(name, x, pemute):
    assert isinstance(x, Node)

    pemute = to_const(pemute, "pemute")

    node = menu.op(name=name, op_name=Name.Layer.transpose, inputs=[x, ])
    node.set(Name.permute, pemute)

    return node


def reshape(name, x, shape):
    assert isinstance(x, Node)

    shape = to_const(shape, "shape")

    node = menu.op(name=name, op_name=Name.Layer.reshape, inputs=[x, ])
    node.set(Name.shape, shape)

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


def conv2d(name, x, w,
           format=Name.NCHW,
           padding=None,
           padding_value=None,
           stride=None,
           dialations=None):
    assert isinstance(x, Node)

    if padding is None:
        padding = Default.padding()
    if padding_value is None:
        padding_value = Default.padding_value()
    if stride is None:
        stride = Default.stride()
    if dialations is None:
        dialations = Default.dialations()
    w = to_node(w, name="_const_" + name + "_weights")

    node = None

    if isinstance(padding, None):
        node = menu.op(name=name, op_name=Name.Layer.padding_conv2d, inputs=[x, padding, w])
        node.set(Name.padding, Default.padding())
    else:
        node = menu.op(name=name, op_name=Name.Layer.conv2d, inputs=[x, w])
        node.set(Name.padding, padding)

    node.set(Name.format, format)
    node.set(Name.padding_value, padding_value)
    node.set(Name.stride, stride)
    node.set(Name.dialations, dialations)

    return node


def shape(name, x):
    assert isinstance(x, Node)

    return menu.op(name=name, op_name=Name.Layer.shape, inputs=[x, ])


def pad(name, x, padding, padding_value=None):
    assert isinstance(x, Node)

    if padding_value is None:
        padding_value = Default.padding_value()
    padding = to_node(padding, name="_const_" + name + "_padding", device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.pad, inputs=[x, padding])
    node.set(Name.padding_value, padding_value)

    return node


def depthwise_conv2d(name, x, w,
                     format=Name.NCHW,
                     padding=None,
                     padding_value=None,
                     stride=None,
                     dialations=None):
    assert isinstance(x, Node)

    if padding is None:
        padding = Default.padding()
    if padding_value is None:
        padding_value = Default.padding_value()
    if stride is None:
        stride = Default.stride()
    if dialations is None:
        dialations = Default.dialations()
    w = to_node(w, name="_const_" + name + "_weights")

    node = None
    if isinstance(padding, None):
        node = menu.op(name=name, op_name=Name.Layer.padding_depthwise_conv2d, inputs=[x, padding, w])
        node.set(Name.padding, Default.padding())
    else:
        node = menu.op(name=name, op_name=Name.Layer.depthwise_conv2d, inputs=[x, w])
        node.set(Name.padding, padding)

    node.set(Name.format, format)
    node.set(Name.padding_value, padding_value)
    node.set(Name.stride, stride)
    node.set(Name.dialations, dialations)

    return node


def add_bias(name, x, b, format=Name.NCHW):
    assert isinstance(x, None)
    assert format == Name.NCHW or format == Name.NHWC

    b = to_node(b, name="_const_" + name + "_bias")

    node = menu.op(name=name, op_name=Name.Layer.add_bias, inputs=[x, b])

    dim = 1
    if format == Name.NCHW:
        dim = 1
    else:
        dim = 3

    node.set(Name.format, format)
    node.set(Name.dim, dim)

    return node


def conv2d_bias(name, x, w, b=None,
                format=Name.NCHW,
                padding=None,
                padding_value=None,
                stride=None,
                dialations=None):
    if b is None:
        return conv2d(name=name, x=x, w=w,
                      format=format,
                      padding=padding, padding_value=padding_value,
                      stride=stride, dialations=dialations)
    else:
        conv_node = conv2d(name="_op_" + name + "_conv2d", x=x, w=w,
                           format=format,
                           padding=padding, padding_value=padding_value,
                           stride=stride, dialations=dialations)
        return add_bias(name=name, x=conv_node, b=b, format=format)


def depthwise_conv2d_bias(name, x, w, b=None,
                          format=Name.NCHW,
                          padding=None,
                          padding_value=None,
                          stride=None,
                          dialations=None):
    if b is None:
        return depthwise_conv2d(name=name, x=x, w=w,
                      format=format,
                      padding=padding, padding_value=padding_value,
                      stride=stride, dialations=dialations)
    else:
        conv_node = depthwise_conv2d(name="_op_" + name + "_conv2d", x=x, w=w,
                           format=format,
                           padding=padding, padding_value=padding_value,
                           stride=stride, dialations=dialations)
        return add_bias(name=name, x=conv_node, b=b, format=format)


def batch_norm(name, x, mean, variance, dim, epsilon):
    assert isinstance(x, None)
    mean = to_node(mean, name="_const_" + name + "_mean")
    variance = to_node(variance, name="_const_" + name + "_variance")

    node = menu.op(name=name, op_name=Name.Layer.batch_norm, inputs=[x, mean, variance])
    node.set(Name.dim, dim)
    node.set(Name.epsilon, epsilon)

    return node


def batch_scale(name, x, scale, bias, dim):
    assert isinstance(x, None)
    scale = to_node(scale, name="_const_" + name + "_mean")
    bias = to_node(bias, name="_const_" + name + "_variance")

    node = menu.op(name=name, op_name=Name.Layer.batch_scale, inputs=[x, scale, bias])
    node.set(Name.dim, dim)

    return node


def fused_batch_norm(name, x, mean, variance, scale, bias, dim, epsilon):
    assert isinstance(x, None)
    mean = to_node(mean, name="_const_" + name + "_mean")
    variance = to_node(variance, name="_const_" + name + "_variance")
    scale = to_node(scale, name="_const_" + name + "_mean")
    bias = to_node(bias, name="_const_" + name + "_variance")

    node = menu.op(name=name, op_name=Name.Layer.fused_batch_norm, inputs=[x, mean, variance, scale, bias])
    node.set(Name.dim, dim)
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


def inner_prod(name, lhs, rhs):
    lhs = to_node(lhs, name="_const_" + name + "_lhs")
    rhs = to_node(rhs, name="_const_" + name + "_rhs")

    node = menu.op(name=name, op_name=Name.Layer.inner_prod, inputs=[lhs, rhs])

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
    node = menu.op(name=name, op_name=Name.Layer.prelu, inputs=[x, ])
    node.set(Name.dim, dim)
    node.set(Name.slope, slope)
    return node


def sigmoid(name, x):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.sigmoid, inputs=[x, ])
    return node


def softmax(name, x, dim):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.softmax, inputs=[x, ])
    node.set(Name.dim, dim)
    return node


def concat(name, inputs, dim):
    for x in inputs:
        assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.concat, inputs=inputs)
    node.set(Name.dim, dim)
    return node


def flatten(name, x):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.flatten, inputs=[x, ])
    return node


def to_float(name, x):
    assert isinstance(x, Node)
    node = menu.op(name=name, op_name=Name.Layer.to_float, inputs=[x, ])
    return node


def resize2d(name, x, size, type=Type.resize2d_type.linear):
    assert isinstance(x, Node)

    size = to_node(size, name="_const_" + name + "_size", device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.resize2d, inputs=[x, size])
    node.set(Name.type, type)

    return node


def pooling2d_v2(name, x, ksize, stride, type=Type.pooling_type.max, format=Name.NCHW,
              padding=None,
              padding_type=Type.padding_type.black):
    assert isinstance(x, Node)

    if padding is None:
        padding = Default.padding()

    padding = to_node(padding, name="_const_" + name + "_padding", device=device.CPU)
    ksize = to_node(ksize, name="_const_" + name + "_ksize", device=device.CPU)
    stride = to_node(stride, name="_const_" + name + "_stride", device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.pooling2d_v2, inputs=[x, padding, ksize, stride])
    node.set(Name.format, format)
    node.set(Name.type, type)
    node.set(Name.padding_type, padding_type)

    return node


def pooling2d(name, x, ksize, stride, type=Type.pooling_type.max, format=Name.NCHW,
              padding=None,
              padding_type=Type.padding_type.black):
    assert isinstance(x, Node)

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
    node.set(Name.padding, padding)
    node.set(Name.ksize, ksize)
    node.set(Name.stride, stride)
    node.set(Name.format, format)
    node.set(Name.type, type)
    node.set(Name.padding_type, padding_type)

    return node





