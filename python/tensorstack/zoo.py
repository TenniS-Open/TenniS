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
        conv2d_bias = "conv2d_bias"
        padding_conv2d_bias = "padding_conv2d_bias"
        shape = "_shape"
        pad = "pad"
        depthwise_conv2d = "depthwise_conv2d"
        padding_depthwise_conv2d = "padding_depthwise_conv2d"
        depthwise_conv2d_bias = "depthwise_conv2d_bias"
        padding_depthwise_conv2d_bias = "padding_depthwise_conv2d_bias"

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


class Default(object):
    @staticmethod
    def padding():
        return [[0, 0], [0, 0], [0, 0], [0, 0]]

    @staticmethod
    def stride():
        return [1, 1, 1, 1]

    @staticmethod
    def dialations():
        return [1, 1, 1, 1]

    @staticmethod
    def padding_value():
        return 0


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
           dialations=None,
           bias=None):
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
    if bias is None:
        if isinstance(padding, None):
            node = menu.op(name=name, op_name=Name.Layer.padding_conv2d, inputs=[x, padding, w])
            node.set(Name.padding, Default.padding())
        else:
            node = menu.op(name=name, op_name=Name.Layer.conv2d, inputs=[x, w])
            node.set(Name.padding, padding)
    else:
        bias = to_node(bias, name="_const_" + name + "_bias")
        if isinstance(padding, None):
            node = menu.op(name=name, op_name=Name.Layer.padding_conv2d_bias, inputs=[x, padding, w, bias])
            node.set(Name.padding, Default.padding())
        else:
            node = menu.op(name=name, op_name=Name.Layer.conv2d_bias, inputs=[x, w, bias])
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
    padding = to_node(padding, name="_const_" +  name + "_padding", device=device.CPU)

    node = menu.op(name=name, op_name=Name.Layer.pad, inputs=[x, padding])
    node.set(Name.padding_value, padding_value)

    return node


def depthwise_conv2d(name, x, w,
                     format=Name.NCHW,
                     padding=None,
                     padding_value=None,
                     stride=None,
                     dialations=None,
                     bias=None):
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
    if bias is None:
        if isinstance(padding, None):
            node = menu.op(name=name, op_name=Name.Layer.padding_depthwise_conv2d, inputs=[x, padding, w])
            node.set(Name.padding, Default.padding())
        else:
            node = menu.op(name=name, op_name=Name.Layer.depthwise_conv2d, inputs=[x, w])
            node.set(Name.padding, padding)
    else:
        bias = to_node(bias, name="_const_" + name + "_bias")
        if isinstance(padding, None):
            node = menu.op(name=name, op_name=Name.Layer.padding_depthwise_conv2d_bias, inputs=[x, padding, w, bias])
            node.set(Name.padding, Default.padding())
        else:
            node = menu.op(name=name, op_name=Name.Layer.depthwise_conv2d_bias, inputs=[x, w, bias])
            node.set(Name.padding, padding)

    node.set(Name.format, format)
    node.set(Name.padding_value, padding_value)
    node.set(Name.stride, stride)
    node.set(Name.dialations, dialations)

    return node


