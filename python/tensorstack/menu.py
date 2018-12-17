#!/usr/bin/env python

"""
:author Kier
"""

from node import Node


def param(name):
    return Node(op=Node.Parameter, name=name)


def op(name, op_name, inputs, output_count=1):
    node = Node(op=op_name, name=name, output_count=output_count)
    Node.Link(node, inputs=inputs)
    return node


def data(name, value):
    node = Node(op=Node.Const, name=name)
    node.set("value", value)
    return node