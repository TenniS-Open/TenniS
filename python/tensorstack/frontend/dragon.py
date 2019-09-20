#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        pooling2d_padding = "_dragon_pooling2d_padding"

    auto_pad = "auto_pad"

    NOTSET = "NOTSET"
    SAME_UPPER = "SAME_UPPER"
    SAME_LOWER = "SAME_LOWER"
    VALID = "VALID"


def proposal(output_names,
             inputs,
             strides,
             ratios,
             scales,
             pre_nms_top_n=6000,
             post_nms_top_n=300,
             nms_thresh=0.7,
             min_size=16,
             min_level=2,
             max_level=5,
             canonical_scale=224,
             canonical_level=4):
    """
    :param output_names: (sequence of string) name of output
    :param inputs: (sequence of Tensor)  The inputs.
    :param strides: (sequence of int)  The strides of anchors.
    :param ratios: (sequence of float)  The ratios of anchors.
    :param scales: (sequence of float)  The scales of anchors.
    :param pre_nms_top_n: (int, optional, default=6000)  The number of anchors before nms.
    :param post_nms_top_n: (int, optional, default=300)  The number of anchors after nms.
    :param nms_thresh: (float, optional, default=0.7)  The threshold of nms.
    :param min_size: (int, optional, default=16)  The min size of anchors.
    :param min_level: (int, optional, default=2)  Finest level of the FPN pyramid.
    :param max_level: (int, optional, default=5)  Coarsest level of the FPN pyramid.
    :param canonical_scale: (int, optional, default=224)  The baseline scale of mapping policy.
    :param canonical_level: (int, optional, default=4)  Heuristic level of the canonical scale.
    :return: The proposals
    """

    for input in inputs:
        assert isinstance(input, Node)

    node = menu.op(name=output_names[0] + "_proposal", op_name="proposal", inputs=inputs)
    node.set("strides", strides, dtype=numpy.int32)
    node.set("ratios", ratios, dtype=numpy.float32)
    node.set("scales", scales, dtype=numpy.float32)

    node.set("pre_nms_top_n", pre_nms_top_n, dtype=numpy.int32)
    node.set("post_nms_top_n", post_nms_top_n, dtype=numpy.int32)
    node.set("nms_thresh", nms_thresh, dtype=numpy.float32)
    node.set("min_size", min_size, dtype=numpy.int32)
    node.set("min_level", min_level, dtype=numpy.int32)
    node.set("max_level", max_level, dtype=numpy.int32)
    node.set("canonical_scale", canonical_scale, dtype=numpy.int32)
    node.set("canonical_level", canonical_level, dtype=numpy.int32)

    return [menu.field(name=output_names[i], input=node, offset=i) for i in range(len(output_names))]


def roi_align(output_names,
              inputs,
              pool_h=0,
              pool_w=0,
              spatial_scale=1.0,
              sampling_ratio=2):
    """
    :param output_names: (sequence of string) name of output
    :param inputs: (sequence of Tensor) – The inputs, represent the Feature and RoIs respectively.
    :param pool_h: (int, optional, default=0) – The height of pooled tensor.
    :param pool_w: (int, optional, default=0) – The width of pooled tensor.
    :param spatial_scale: (float, optional, default=1.0) – The inverse of total down-sampling multiples on input tensor.
    :param sampling_ratio: (int, optional, default=2) – The number of sampling grids for each RoI bin.
    :return:
    """

    for input in inputs:
        assert isinstance(input, Node)

    node = menu.op(name=output_names[0] + "_roi_align", op_name="roi_align", inputs=inputs)
    node.set("pool_h", pool_h, dtype=numpy.int32)
    node.set("pool_w", pool_w, dtype=numpy.int32)

    node.set("spatial_scale", spatial_scale, dtype=numpy.float32)
    node.set("sampling_ratio", sampling_ratio, dtype=numpy.int32)

    return [menu.field(name=output_names[i], input=node, offset=i) for i in range(len(output_names))]


def pooling2d_padding(name, x, padding, ksize, stride, auto_pad=Name.NOTSET, ceil_mode=True):
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
    node.set("ceil", ceil_mode, numpy.bool)

    return node


def pooling2d(name, x, ksize, stride, type=zoo.Type.pooling_type.max, format=zoo.Name.NCHW,
              padding=None,
              padding_type=zoo.Type.padding_type.black,
              auto_pad=Name.VALID,
              ceil_mode=None):
    assert isinstance(x, Node)

    if ceil_mode is None:
        ceil_mode = True

    padding = zoo.adjust_padding(padding, format=format)
    ksize = zoo.adjust_ksize(ksize, format=format)
    stride = zoo.adjust_stride(stride, format=format)

    if format != zoo.Name.NCHW:
        raise NotImplementedError("Dragon format = {}".format(format))

    if padding is None:
        padding = zoo.Default.padding()
    # param
    static_padding = zoo.to_const(padding, "padding")

    # input
    ksize = zoo.to_node(ksize, name="_const_" + name + "_ksize", dtype=numpy.int32, device=device.CPU)
    stride = zoo.to_node(stride, name="_const_" + name + "_stride", dtype=numpy.int32, device=device.CPU)

    # operator
    dynamic_padding = pooling2d_padding(name="_op_" + name + "_dragon_padding",
                                        x=x, padding=static_padding, ksize=ksize, stride=stride, auto_pad=auto_pad, ceil_mode=ceil_mode)

    return zoo.pooling2d_v2(name=name, x=x, ksize=ksize, stride=stride,
                            type=type, format=format, padding=dynamic_padding, padding_type=padding_type)
