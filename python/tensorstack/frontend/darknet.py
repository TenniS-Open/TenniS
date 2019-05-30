#!/usr/bin/env python

"""
Author: Kier
"""

from .. import Node
from .. import zoo
from .. import menu
from .. import device

import numpy


def yolo(name, x, classes, mask, anchors):
    node = menu.op(name=name, op_name="yolo", inputs=[x])

    node.set("classes", classes, numpy.int32)
    node.set("mask", mask, numpy.int32)
    node.set("anchors", anchors, numpy.float32)

    return node


def yolo_poster(name, x, inputs, thresh, nms):
    assert isinstance(x, Node)
    assert isinstance(inputs, (Node, tuple, list))
    if isinstance(inputs, Node):
        inputs = [inputs,]
    for yolo in inputs:
        assert isinstance(yolo, Node)
        assert yolo.op == "yolo"

    x = [x,]
    x.extend(inputs)

    node = menu.op(name=name, op_name="yolo_poster", inputs=x)
    node.set("thresh", thresh, numpy.float32)
    node.set("nms", nms, numpy.float32)

    return node
