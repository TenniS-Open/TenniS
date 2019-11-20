#!/usr/bin/env python

"""
Author: Kier
"""

from .. import Node
from .. import zoo
from .. import menu
from .. import device

import numpy


class Name(object):
    class Layer(object):
        pooling2d_padding = "_onnx_pooling2d_padding"
        gather = "gather"
        unsqueeze = "unsqueeze"
        gemm = "gemm"

    auto_pad = "auto_pad"
    axis = "axis"
    axes = "axes"

    NOTSET = "NOTSET"
    SAME_UPPER = "SAME_UPPER"
    SAME_LOWER = "SAME_LOWER"
    VALID = "VALID"

    alpha = "alpha"
    beta = "beta"
    transA = "transA"
    transB = "transB"


def pooling2d_padding(name, x, padding, ksize, stride, auto_pad=Name.NOTSET):
    assert isinstance(x, Node)
    assert auto_pad in {Name.NOTSET, Name.SAME_LOWER, Name.SAME_UPPER, Name.VALID}

    padding = zoo.adjust_padding(padding, format=zoo.Name.NCHW)
    ksize = zoo.adjust_ksize(ksize, format=zoo.Name.NCHW)
    stride = zoo.adjust_stride(stride, format=zoo.Name.NCHW)

    if auto_pad not in {Name.NOTSET, Name.SAME_UPPER, Name.SAME_LOWER, Name.VALID}:
        raise NotImplementedError("auto_pad = {}".format(auto_pad))

    # param
    padding = zoo.to_const(padding, "padding")

    # input
    ksize = zoo.to_node(ksize, name="_const_" + name + "_ksize", dtype=numpy.int32, device=device.CPU)
    stride = zoo.to_node(stride, name="_const_" + name + "_stride", dtype=numpy.int32, device=device.CPU)

    # operator
    node = menu.op(name=name, op_name=Name.Layer.pooling2d_padding, inputs=[x, ksize, stride])
    node.set(zoo.Name.padding, padding, numpy.int32)
    node.set(Name.auto_pad, auto_pad)

    return node


def pooling2d(name, x, ksize, stride, type=zoo.Type.pooling_type.max, format=zoo.Name.NCHW,
              padding=None,
              padding_type=zoo.Type.padding_type.black,
              auto_pad=Name.NOTSET,
              ceil_mode=False):
    assert isinstance(x, Node)

    if ceil_mode:
        return zoo.pooling2d(name=name, x=x, ksize=ksize, stride=stride, type=type,
                             format=format, padding=padding, padding_type=padding_type)

    padding = zoo.adjust_padding(padding, format=format)
    ksize = zoo.adjust_ksize(ksize, format=format)
    stride = zoo.adjust_stride(stride, format=format)

    if format != zoo.Name.NCHW:
        raise NotImplementedError("ONNX format = {}".format(format))

    if padding is None:
        padding = zoo.Default.padding()
    # param
    static_padding = zoo.to_const(padding, "padding")

    # input
    ksize = zoo.to_node(ksize, name="_const_" + name + "_ksize", dtype=numpy.int32, device=device.CPU)
    stride = zoo.to_node(stride, name="_const_" + name + "_stride", dtype=numpy.int32, device=device.CPU)

    # operator
    dynamic_padding = pooling2d_padding(name="_op_" + name + "_onnx_padding",
                                        x=x, padding=static_padding, ksize=ksize, stride=stride, auto_pad=auto_pad)

    return zoo.pooling2d_v2(name=name, x=x, ksize=ksize, stride=stride,
                            type=type, format=format, padding=dynamic_padding, padding_type=padding_type)


def gather(name, x, indices, axis=0):
    assert isinstance(x, Node)

    indices = zoo.to_node(indices, name="_const_" + name + "_indices", dtype=numpy.int32, device=device.CPU)
    axis = zoo.to_const(axis, "axis")

    # operator
    node = menu.op(name=name, op_name=Name.Layer.gather, inputs=[x, indices])
    node.set(Name.axis, axis, numpy.int32)

    return node


def unsqueeze(name, x, axes):
    assert isinstance(x, Node)

    axes = zoo.to_const(axes, "axes")

    # operator
    node = menu.op(name=name, op_name=Name.Layer.unsqueeze, inputs=[x, ])
    node.set(Name.axes, axes, numpy.int32)

    return node


def gemm(name, A, B, C, alpha, beta, transA, transB):
    A = zoo.to_node(value=A, name="A")
    B = zoo.to_node(value=B, name="B")
    C = zoo.to_node(value=C, name="C")

    alpha = zoo.to_const(value=alpha, name="alpha")
    beta = zoo.to_const(value=beta, name="beta")
    transA = zoo.to_const(value=transA, name="transA")
    transB = zoo.to_const(value=transB, name="transB")

    # operator
    node = menu.op(name=name, op_name=Name.Layer.gemm, inputs=[A, B, C])
    node.set(Name.alpha, alpha, numpy.float32)
    node.set(Name.beta, beta, numpy.float32)
    node.set(Name.transA, transA, numpy.uint8)
    node.set(Name.transB, transB, numpy.uint8)

    return node


def slice_v3(name, x, starts, ends, axes=None, steps=None):
    if steps is not None and axes is None:
        raise ValueError("axes must be set, if steps set.")

    node_inputs = []
    if axes is None:
        node_inputs = [x, starts, ends]
    elif steps is None:
        node_inputs = [x, starts, ends, axes]
    else:
        node_inputs = [x, starts, ends, axes, steps]

    node = menu.op(name=name, op_name="slice_v3", inputs=node_inputs)

    return node

    pass

