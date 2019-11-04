#!/usr/bin/env python

"""
:author Kier
"""

from typing import Union, List, Dict

from .node import Node
from . import dtype as ts_dtype
from .dtype import VOID, FLOAT32
from . import tensor
from . import menu

import sys


class NodeShape(object):
    def __init__(self, obj=None, dtype=None):
        # type: (Union[Node, List[int], tuple[int]], int) -> None
        shape = obj
        if isinstance(obj, Node):
            shape = obj.try_get(Node.RetentionParam.shape, None)
            if shape is None:
                raise Exception("Param 1 Node {}:{}'s shape must be set.".format(obj.op, obj.name))
        elif isinstance(obj, (list, tuple)):
            shape = obj
        else:
            raise Exception("Param 1:obj must be Node, List[int] or List[tuple]")
        if dtype is None:
            dtype = FLOAT32
        self.shape = shape
        if not isinstance(dtype, int):
            dtype = ts_dtype.from_numpy(dtype=dtype)
        self.dtype = dtype

    def empty(self):
        return self.dtype == VOID

    def __str__(self):
        return "{}:{}".format(ts_dtype.dtype_str(self.dtype), self.shape)

    def __repr__(self):
        return "{}:{}".format(ts_dtype.dtype_str(self.dtype), self.shape)


def infer_param(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[NodeShape, List[NodeShape]]
    return NodeShape(node)


def infer_data(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[NodeShape, List[NodeShape]]
    value = node.get("value")
    value = tensor.from_any(value)
    return NodeShape(value.shape, value.dtype)


class _inferer(object):
    def __call__(self, node, inputs):
        # type: (Node, List[NodeShape]) -> Union[NodeShape, List[NodeShape]]
        raise NotImplementedError


_shape_inferer = {
    Node.Parameter: infer_param,
    Node.Const: infer_data,
}


def _register_shape_inferer(op, inferer):
    _shape_inferer[op] = inferer


def _has_infered(node):
    # type: (Node) -> bool
    return node.has(Node.RetentionParam.shape) and node.has(Node.RetentionParam.dtype)


def infer(node, cache=None):
    # type: (Union[Node, List[Node]], Dict[Node, List[NodeShape]]) -> Union[List[NodeShape], List[List[NodeShape]]]
    """
    :param node:
    :return: shape
    """
    if cache is None:
        cache = {}

    if isinstance(node, (list, tuple)):
        return [infer(n, cache) for n in node]

    if node in cache:
        return cache[node]

    inputs = [infer(i, cache) for i in node.inputs]
    if node.op not in _shape_inferer:
        sys.stderr.write("Failed infer shape of {}:{}\n".format(node.op, node.name))
        node.dtype = VOID
        node.shape = [-1]
        cache[node] = []
        return []

    for i in inputs:
        if len(i) == 0:
            node.dtype = VOID
            node.shape = [-1]
            cache[node] = []
            return []

    inputs = [i[0] for i in inputs]
    shape = _shape_inferer[node.op](node, inputs)

    if isinstance(shape, NodeShape):
        shape = [shape]

    if shape is None or len(shape) == 0:
        node.dtype = VOID
        node.shape = [-1]
        cache[node] = []
        return []

    node.shape = shape[0].shape
    node.dtype = shape[0].dtype
    cache[node] = shape
    return shape


def infer_add(node, inputs):
    # type: (Node, List[NodeShape]) -> NodeShape
    assert len(inputs) == 2
    import numpy
    a = inputs[0]
    b = inputs[1]
    c = numpy.zeros(a.shape) + numpy.zeros(b.shape)
    return NodeShape(c.shape, c.dtype)


_register_shape_inferer("add", infer_add)


if __name__ == "__main__":
    a = menu.param("a", [3], FLOAT32)
    b = menu.param("b", [3], FLOAT32)
    c = menu.op("c", "add", [a, b])
    d = menu.op("d", "add", [a, c])

    print(infer(d))