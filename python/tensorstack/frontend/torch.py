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
        dcn_v2_forward = "dcn_v2_forward"

    deformable_groups = "deformable_groups"


def dcn_v2_forward(name, x, w, b, offset, mask,
           deformable_groups,
           format=zoo.Name.NCHW,
           padding=None,
           stride=None,
           dilation=None):
    assert isinstance(x, Node)

    padding = zoo.adjust_padding(padding, format=format)
    stride = zoo.adjust_stride(stride, format=format)
    dilation = zoo.adjust_dilation(dilation, format=format)

    if padding is None:
        padding = zoo.Default.padding()
    if stride is None:
        stride = zoo.Default.stride()
    if dilation is None:
        dilation = zoo.Default.dilation()
    w = zoo.to_node(w, name="_const_" + name + "_weights")
    b = zoo.to_node(b, name="_const_" + name + "_bias")
    offset = zoo.to_node(offset, name="_const_" + name + "_offset")
    mask = zoo.to_node(mask, name="_const_" + name + "_mask")

    node = None

    node = menu.op(name=name, op_name=Name.Layer.dcn_v2_forward, inputs=[x, w, b, offset, mask])
    node.set(zoo.Name.padding, padding, numpy.int32)

    node.set(zoo.Name.format, format)
    node.set(Name.deformable_groups, deformable_groups, numpy.int32)
    node.set(zoo.Name.stride, stride, numpy.int32)
    node.set(zoo.Name.dilation, dilation, numpy.int32)
    # the kernel size is w.shape[2:4]

    return node