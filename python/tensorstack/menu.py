#!/usr/bin/env python

"""
:author Kier
"""

from node import Node


class Name(object):
    class Layer(object):
        field = "_field"
        pack = "_pack"

    offset = "offset"
    value = "value"
    device = "device"


def param(name):
    # type: (str) -> Node
    return Node(op=Node.Parameter, name=name)


def op(name, op_name, inputs, output_count=1):
    # type: (str, str, list[Node], int) -> Node
    node = Node(op=op_name, name=name, output_count=output_count)
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
    node.set(Name.offset, offset)
    Node.Link(node=node, inputs=[input])
    return node


def pack(name, inputs):
    # type: (str, list[Node]) -> Node
    node = Node(op=Name.Layer.pack, name=name)
    Node.Link(node=node, inputs=inputs)
    return node
