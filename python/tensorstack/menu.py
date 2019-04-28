#!/usr/bin/env python

"""
:author Kier
"""

from .node import Node
from .dtype import from_numpy
import numpy


class Name(object):
    class Layer(object):
        field = "_field"
        pack = "_pack"

    offset = "offset"
    value = "value"
    device = "device"


def param(name, shape=None, dtype=None):
    # type: (str, list[int], object) -> Node
    node = Node(op=Node.Parameter, name=name, shape=shape)
    if dtype is not None:
        if not isinstance(dtype, int):
            dtype = from_numpy(dtype)
        node.set(Node.RetentionParam.dtype, dtype, numpy.int32)
    return node


def op(name, op_name, inputs, output_count=1):
    # type: (str, str, list[Node], int) -> Node
    if output_count != 1:
        raise Exception("output_count must be 1.")
    node = Node(op=op_name, name=name)
    Node.Link(node, inputs=inputs)
    return node


def data(name, value, device=None):
    # type: (str, object, Union[str]) -> Node
    node = Node(op=Node.Const, name=name)
    node.set(Name.value, value)
    if device is not None:
        node.set(Name.device, device)
    return node


def field(name, input, offset):
    # type: (str, Node, int) -> Node
    node = Node(op=Name.Layer.field, name=name)
    node.set(Name.offset, offset, numpy.int32)
    Node.Link(node=node, inputs=[input])
    return node


def pack(name, inputs):
    # type: (str, list[Node]) -> Node
    node = Node(op=Name.Layer.pack, name=name)
    Node.Link(node=node, inputs=inputs)
    return node
