#!python
# coding: UTF-8
"""
author: kier
"""

from ..enum import *
from ..config import *
from ..darknet import Layer
from ..darknet import Network

import tensorstack as ts
import numpy


def __fused_batch_norm(w, b, mean, variance, epsilon=1e-5, w_shape=None):
    """
    Notice y = (x - mean) / (sqrt(var) + eps), not same as common BN
    :param w:
    :param b:
    :param mean:
    :param variance:
    :param epsilon:
    :param w_shape:
    :return:
    """
    if w_shape is None:
        w_shape = (variance.shape[0], 1, 1, 1)
    d = variance + epsilon
    std = numpy.sqrt(variance) + epsilon
    w = w / numpy.reshape(std, newshape=w_shape)
    b = (b - mean) / std
    return w, b


def __fused_batch_scale(w, b, scale, bias, w_shape=None):
    if w_shape is None:
        w_shape = (scale.shape[0], 1, 1, 1)
    w = w * numpy.reshape(scale, newshape=w_shape)
    b = b * scale + bias
    return w, b


def activate_array(x, shape, a, name=None):
    # type: (ts.Node, Union[tuple, list], int, str) -> ts.Node
    if name is None:
        name = x.name + "_act"

    map_activation = {
        RELU: ts.zoo.relu,
        LEAKY: lambda name, x: ts.zoo.prelu(name=name, x=x, dim=1,
                                            slope=numpy.asarray([0.1] * shape[1], dtype=numpy.float32)),
        LINEAR: ts.zoo.copy,
    }

    if a not in map_activation:
        a_str = ACTIVATION_STRING[a] if a in ACTIVATION_STRING else "unknown"
        raise Exception("Activation = {}".format(a_str))

    x = map_activation[a](name=name, x=x)

    return x


def forward_convolutional_layer(l, net):
    # type: (Layer, Network) -> ts.Node
    if l.xnor:
        raise NotImplementedError("Not supporting convert with xnor=1")

    weights = l.weights # # l.n, l.c, l.size, l.size
    biases = l.biases

    if l.groups != 1 and l.groups != l.c:
        raise NotImplementedError("Not supporting group=".format(l.groups))

    assert isinstance(weights, numpy.ndarray)
    assert isinstance(biases, numpy.ndarray)

    weights = numpy.reshape(weights, newshape=(l.n // l.groups, l.c, l.size, l.size))

    is_depthwise = weights.shape[0] == 1

    if l.batch_normalize:
        # fuse bn to conv
        if is_depthwise:
            w_shape = (weights.shape[0], weights.shape[1], 1, 1)
        else:
            w_shape = (weights.shape[0], 1, 1, 1)

        mean = l.rolling_mean
        variance = l.rolling_variance
        scales = l.scales

        w, b = weights, numpy.zeros_like(scales)
        w, b = __fused_batch_norm(w=w, b=b, mean=mean, variance=variance, epsilon=0.000001,
                                  w_shape=w_shape)
        w, b = __fused_batch_scale(w=w, b=b, scale=scales, bias=biases,
                                   w_shape=w_shape)
        weights, biases = w, b

    layer_name = str(net.index)

    x = net.input
    if is_depthwise:
        x = ts.zoo.depthwise_conv2d(name=layer_name + "_conv", x=x, w=weights,
                                    padding=l.pad, stride=l.stride)
    else:
        x = ts.zoo.conv2d(name=layer_name + "_conv", x=x, w=weights,
                          padding=l.pad, stride=l.stride)

    x = ts.zoo.add_bias(name=layer_name + "_bias", x=x, b=biases)

    x = activate_array(x, (l.batch, l.out_c, l.out_h, l.out_w), l.activation, name=layer_name)

    l.output = x
    return x


def forward_maxpool_layer(l, net):
    # type: (Layer, Network) -> ts.Node

    layer_name = str(net.index)

    x = net.input

    padding = [
        [l.pad // 2, l.pad - l.pad // 2],
        [l.pad // 2, l.pad - l.pad // 2],
    ]

    x = ts.frontend.onnx.pooling2d(name=layer_name, x=x, ksize=l.size, stride=l.stride,
                                   type=ts.zoo.Type.pooling_type.max,
                                   padding=padding, padding_type=ts.zoo.Type.padding_type.black,
                                   auto_pad="NOTSET")

    l.output = x
    return x


def forward_yolo_layer(l, net):
    # type: (Layer, Network) -> ts.Node

    layer_name = str(net.index)

    x = net.input
    x = ts.frontend.yolo(name=layer_name, x=x, classes=l.classes, mask=l.mask, anchors=l.biases)

    l.output = x
    return x


def forward_route_layer(l, net):
    # type: (Layer, Network) -> ts.Node

    layer_name = str(net.index)

    input_nodes = [net.layers[i].output for i in l.input_layers]

    if len(input_nodes) == 1:
        x = ts.zoo.copy(name=layer_name, x=input_nodes[0])
    else:
        x = ts.zoo.concat(name=layer_name, inputs=input_nodes, dim=1)

    l.output = x
    return x


def forward_upsample_layer(l, net):
    # type: (Layer, Network) -> ts.Node

    layer_name = str(net.index)

    if l.has("reverse") and l.reverse:
        raise NotImplementedError("Not support with reverse=1")

    x = net.input

    x = ts.zoo.sample2d(name=layer_name + "_upsample", x=x, scale=l.stride,
                        type=ts.zoo.Type.resize2d_type.hard, dim=-2)

    if l.scale != 1:
        x = ts.zoo.mul(name=layer_name + "_scale", lhs=x, rhs=ts.tensor.from_any(l.scale, numpy.float32))

    x.name = layer_name

    l.output = x
    return x


def forward_shortcut_layer(l, net):
    # type: (Layer, Network) -> ts.Node
    layer_name = str(net.index)

    lhs = net.input
    rhs = net.layers[l.index].output

    scale_lhs = l.alpha
    scale_rhs = l.beta

    if scale_lhs != 1:
        lhs = ts.zoo.mul(name=layer_name + "_alpha_lhs", lhs=lhs, rhs=numpy.asarray(scale_lhs, dtype=numpy.float32))

    if scale_rhs != 1:
        rhs = ts.zoo.mul(name=layer_name + "_beta_rhs", lhs=rhs, rhs=numpy.asarray(scale_rhs, dtype=numpy.float32))

    x = ts.zoo.add(name=layer_name + "_add", lhs=lhs, rhs=rhs)

    x = activate_array(x, (l.batch, l.out_c, l.out_h, l.out_w), l.activation, name=layer_name)

    l.output = x
    return x
