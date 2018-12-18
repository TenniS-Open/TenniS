#!/usr/bin/env python

"""
:author Kier
"""

from node import Node


def param(name):
    # type: (str) -> Node
    return Node(op=Node.Parameter, name=name)


def op(name, op_name, inputs, output_count=1):
    # type: (str, str, list[Node], int) -> Node
    node = Node(op=op_name, name=name, output_count=output_count)
    Node.Link(node, inputs=inputs)
    return node


def data(name, value):
    # type: (str, object) -> Node
    node = Node(op=Node.Const, name=name)
    node.set("value", value)
    return node


def field(name, input, offset):
    # type: (str, Node, int) -> Node
    node = Node(op="_field", name=name)
    node.set("offset", offset)
    Node.Link(node=node, inputs=[input])
    return node


def pack(name, inputs):
    # type: (str, list[Node]) -> Node
    node = Node(op="_pack", name=name)
    Node.Link(node=node, inputs=inputs)
    return node
