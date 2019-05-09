#!/usr/bin/env python

"""
Author: Kier
"""

from .. import Node
from .. import zoo
from .. import menu
from .. import device
from .. import tensor

import numpy


class Name(object):
    class Layer(object):
        conv2d_padding = "_tf_conv2d_padding"
        pooling2d_padding = "_tf_pooling2d_padding"
        strided_slice = "strided_slice"
        stack = "stack" # use for pack operator
        mean = "mean"
        space_to_batch4d = "space_to_batch4d"
        batch_to_space4d = "batch_to_space4d"

    SAME = "SAME"
    VALID = "VALID"

    begin = "begin"
    end = "end"
    stride = "stride"
    axis = "axis"

    padding_method = "padding_method"

    block_shape = "block_shape"
    crop = "crop"


def pooling2d_padding(name, x, padding, ksize, stride, format=zoo.Name.NCHW, padding_method=Name.SAME):
    assert isinstance(x, Node)

    padding = zoo.adjust_padding(padding, format=format)
    ksize = zoo.adjust_ksize(ksize, format=format)
    stride = zoo.adjust_stride(stride, format=format)

    if padding_method not in {Name.SAME, Name.VALID}:
        raise NotImplementedError("padding_method = {}".format(padding_method))

    if format not in {zoo.Name.NCHW, zoo.Name.NHWC}:
        raise NotImplementedError("format = {}".format(format))

    # param
    padding = zoo.to_const(padding, "padding")

    # input
    ksize = zoo.to_node(ksize, name="_const_" + name + "_ksize", dtype=numpy.int32, device=device.CPU)
    stride = zoo.to_node(stride, name="_const_" + name + "_stride", dtype=numpy.int32, device=device.CPU)

    # operator
    node = menu.op(name=name, op_name=Name.Layer.pooling2d_padding, inputs=[x, ksize, stride])
    node.set(zoo.Name.padding, padding, numpy.int32)
    node.set(Name.padding_method, padding_method)
    node.set(zoo.Name.format, format)

    return node


def pooling2d(name, x, ksize, stride, type=zoo.Type.pooling_type.max, format=zoo.Name.NCHW,
              padding=None,
              padding_type=zoo.Type.padding_type.black,
              padding_method=Name.SAME):
    assert isinstance(x, Node)

    padding = zoo.adjust_padding(padding, format=format)
    ksize = zoo.adjust_ksize(ksize, format=format)
    stride = zoo.adjust_stride(stride, format=format)

    if padding is None:
        padding = zoo.Default.padding()

    # param
    static_padding = zoo.to_const(padding, "padding")

    # input
    ksize = zoo.to_node(ksize, name="_const_" + name + "_ksize", dtype=numpy.int32, device=device.CPU)
    stride = zoo.to_node(stride, name="_const_" + name + "_stride", dtype=numpy.int32, device=device.CPU)

    # operator
    dynamic_padding = pooling2d_padding(name="_op_" + name + "_tf_padding",
                                        x=x, padding=static_padding, ksize=ksize, stride=stride,
                                        format=format,
                                        padding_method=padding_method)

    return zoo.pooling2d_v2(name=name, x=x, ksize=ksize, stride=stride,
                            type=type, format=format, padding=dynamic_padding, padding_type=padding_type)


def conv2d_padding(name, x, w,
                   format=zoo.Name.NCHW,
                   padding=None,
                   padding_method=Name.SAME,
                   stride=None,
                   dilation=None):
    assert isinstance(x, Node)

    padding = zoo.adjust_padding(padding, format=format)
    stride = zoo.adjust_stride(stride, format=format)
    dilation = zoo.adjust_dilation(dilation, format=format)

    if padding_method not in {Name.SAME, Name.VALID}:
        raise NotImplementedError("padding_method = {}".format(padding_method))

    if format not in {zoo.Name.NCHW, zoo.Name.NHWC}:
        raise NotImplementedError("format = {}".format(format))

    if padding is None:
        padding = zoo.Default.padding()
    if stride is None:
        stride = zoo.Default.stride()
    if dilation is None:
        dilation = zoo.Default.dilation()
    w = zoo.to_node(w, name="_const_" + name + "_weights")

    padding = tensor.from_any(padding, numpy.int32)
    assert padding.shape == (4, 2)

    node = menu.op(name=name, op_name=Name.Layer.conv2d_padding, inputs=[x, w])
    node.set(zoo.Name.padding, padding, numpy.int32)
    node.set(zoo.Name.format, format)
    node.set(Name.padding_method, padding_method)
    node.set(zoo.Name.stride, stride, numpy.int32)
    node.set(zoo.Name.dilation, dilation, numpy.int32)

    return node


def conv2d(name, x, w,
           format=zoo.Name.NCHW,
           padding=None,
           padding_method=Name.SAME,
           padding_value=None,
           stride=None,
           dilation=None):
    assert isinstance(x, Node)

    padding = zoo.adjust_padding(padding, format=format)
    stride = zoo.adjust_stride(stride, format=format)
    dilation = zoo.adjust_dilation(dilation, format=format)

    if padding_method not in {Name.SAME, Name.VALID}:
        raise NotImplementedError("padding_method = {}".format(padding_method))

    if format not in {zoo.Name.NCHW, zoo.Name.NHWC}:
        raise NotImplementedError("format = {}".format(format))

    if padding is None:
        padding = zoo.Default.padding()
    if padding_value is None:
        padding_value = zoo.Default.padding_value()
    if stride is None:
        stride = zoo.Default.stride()
    if dilation is None:
        dilation = zoo.Default.dilation()
    w = zoo.to_node(w, name="_const_" + name + "_weights")

    # operator
    dynamic_padding = conv2d_padding(name="_op_" + name + "_tf_padding",
                                     x=x, w=w, format=format, padding=padding, padding_method=padding_method,
                                     stride=stride, dilation=dilation)

    return zoo.conv2d(name=name, x=x, w=w, format=format, padding=dynamic_padding, padding_value=padding_value,
                      stride=stride, dilation=dilation)


def strided_slice(name, x, begin, end, stride=None):
    """
    return x in [begin, end) with stride
    :param name:
    :param x:
    :param begin:
    :param end:
    :param stride:
    :return:
    """
    assert isinstance(x, Node)

    if stride is None:
        stride = [1, ] * len(begin)

    begin = zoo.to_const(begin, "begin")
    end = zoo.to_const(end, "end")
    stride = zoo.to_const(stride, "stride")

    assert len(begin) == len(end)
    assert len(begin) == len(stride)

    # operator
    node = menu.op(name=name, op_name=Name.Layer.strided_slice, inputs=[x, ])
    node.set(Name.begin, begin, numpy.int32)
    node.set(Name.end, end, numpy.int32)
    node.set(Name.stride, stride, numpy.int32)

    return node


def stack(name, tensors, axis=0):
    """
    tf.concat(axis, [tf.expand_dims(t, axis) for t in tensors])
    IS
    tf.pack(tensors, axis=axis)
    :param name:
    :param x:
    :param axis:
    :return:
    """

    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors, ]

    axis = zoo.to_const(axis, "axis")

    node = menu.op(name=name, op_name=Name.Layer.stack, inputs=tensors)
    node.set(Name.axis, axis, numpy.int32)

    return node


def mean(name, x, w=None):
    isinstance(x, Node)
    if w is None:
        return menu.op(name=name, op_name=Name.Layer.mean, inputs=[x, ])

    w = zoo.to_node(w, name + "_w")
    return menu.op(name=name, op_name=Name.Layer.mean, inputs=[x, w])


def space_to_batch4d(name, x, block_shape, padding):
    assert isinstance(x, Node)

    block_shape = zoo.to_const(block_shape, "block_shape")
    padding = zoo.to_const(padding, "padding")

    block_shape = tensor.from_any(block_shape, dtype=numpy.int32)
    padding = tensor.from_any(padding, dtype=numpy.int32)

    if block_shape.shape != (2,):
        raise NotImplementedError("block_shape.shape must be [2], got {}".format(block_shape))

    if padding.shape != (2, 2):
        raise NotImplementedError("padding.shape must be [2, 2], got {}".format(padding))

    node = menu.op(name=name, op_name=Name.Layer.space_to_batch4d, inputs=[x,])
    node.set(Name.block_shape, block_shape, numpy.int32)
    node.set(zoo.Name.padding, padding, numpy.int32)

    return node


def batch_to_space4d(name, x, block_shape, crop):
    assert isinstance(x, Node)

    block_shape = zoo.to_const(block_shape, "block_shape")
    crop = zoo.to_const(crop, "crop")

    block_shape = tensor.from_any(block_shape, dtype=numpy.int32)
    crop = tensor.from_any(crop, dtype=numpy.int32)

    if block_shape.shape != (2,):
        raise NotImplementedError("block_shape.shape must be [2], got {}".format(block_shape))

    if crop.shape != (2, 2):
        raise NotImplementedError("crop.shape must be [2, 2], got {}".format(crop))

    node = menu.op(name=name, op_name=Name.Layer.batch_to_space4d, inputs=[x,])
    node.set(Name.block_shape, block_shape, numpy.int32)
    node.set(Name.crop, crop, numpy.int32)

    return node