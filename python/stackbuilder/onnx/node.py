#!/usr/bin/env python

"""
Author: Kier
"""

from tensorstack import Node
from tensorstack import zoo
from tensorstack import menu
from tensorstack import device

import numpy


class Name(object):
    class Layer(object):
        pooling2d_padding = "_onnx_pooling2d_padding"

    auto_pad = "auto_pad"

    NOTSET = "NOTSET"
    SAME_UPPER = "SAME_UPPER"
    SAME_LOWER = "SAME_LOWER"
    VALID = "VALID"


def pooling2d_padding(name, x, padding, ksize, stride, auto_pad=Name.NOTSET):
    assert isinstance(x, Node)

    if auto_pad not in {Name.NOTSET, Name.SAME_UPPER, Name.SAME_LOWER, Name.VALID}:
        raise NotImplementedError("auto_pad = {}".format(auto_pad))

    # param
    padding = zoo.to_const(padding, "padding")

    # input
    ksize = zoo.to_node(ksize, name="_const_" + name + "_ksize", device=device.CPU)
    stride = zoo.to_node(stride, name="_const_" + name + "_stride", device=device.CPU)

    # operator
    node = menu.op(name=name, op_name=Name.Layer.pooling2d_padding, inputs=[x, ksize, stride])
    node.set(zoo.Name.padding, padding, numpy.int32)
    node.set(Name.auto_pad, auto_pad)

    return node


def pooling2d(name, x, ksize, stride, type=zoo.Type.pooling_type.max, format=zoo.Name.NCHW,
              padding=None,
              padding_type=zoo.Type.padding_type.black,
              auto_pad=Name.NOTSET):
    assert isinstance(x, Node)

    if format != zoo.Name.NCHW:
        raise NotImplementedError("ONNX format = {}".format(format))

    # param
    static_padding = zoo.to_const(padding, "padding")

    # input
    ksize = zoo.to_node(ksize, name="_const_" + name + "_ksize", device=device.CPU)
    stride = zoo.to_node(stride, name="_const_" + name + "_stride", device=device.CPU)

    # operator
    dynamic_padding = pooling2d_padding(name="_op_" + name + "_valid_padding",
                                        x=x, padding=static_padding, ksize=ksize, stride=stride, auto_pad=auto_pad)

    return zoo.pooling2d_v2(name=name, x=x, ksize=ksize, stride=stride,
                            type=type, format=format, padding=dynamic_padding, padding_type=padding_type)
