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
import copy
import math
import numpy


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


def has_infered(node):
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
        sys.stderr.write("Failed infer shape of {}:{}\n".format(node.op, node.name))
        node.dtype = VOID
        node.shape = [-1]
        cache[node] = []
        return []

    node.shape = shape[0].shape
    node.dtype = shape[0].dtype
    cache[node] = shape
    return shape


def infer_eltwise(node, inputs):
    # type: (Node, List[NodeShape]) -> NodeShape
    assert len(inputs) == 2
    import numpy
    a = inputs[0]
    b = inputs[1]
    c = numpy.zeros(a.shape) + numpy.zeros(b.shape)
    return NodeShape(c.shape, c.dtype)


_register_shape_inferer("add", infer_eltwise)
_register_shape_inferer("sub", infer_eltwise)
_register_shape_inferer("mul", infer_eltwise)
_register_shape_inferer("div", infer_eltwise)


def infer_to_float(node, inputs):
    # type: (Node, List[NodeShape]) -> NodeShape
    assert len(inputs) == 1
    x = inputs[0]
    return NodeShape(x.shape, FLOAT32)


_register_shape_inferer("to_float", infer_to_float)


def infer_resize2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2
    x = inputs[0]
    size = node.inputs[1]
    if size.op != Node.Const:
        return None
    size = size.get("value")
    y = list(x.shape)
    if len(y) != len(size):
        return None
    for i in range(len(y)):
        if size[i] > 0:
            y[i] = size[i]
    return NodeShape(y, x.dtype)


_register_shape_inferer("_resize2d", infer_resize2d)


def infer_transpose(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1
    x = inputs[0]
    permute = node.try_get("permute", None)
    if permute is None:
        permute = [i for i in range(len(x.shape))]
        permute[-1], permute[-2] = permute[-2], permute[1]
    y = [0] * len(x.shape)
    for i in range(len(y)):
        y[i] = x.shape[permute[i]]

    return NodeShape(y, x.dtype)


_register_shape_inferer("_transpose", infer_transpose)


def infer_crop_nd(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2
    x = inputs[0]
    size = node.inputs[1]
    if size.op != Node.Const:
        return None
    size = size.get("value")
    y = list(x.shape)
    if len(y) != len(size):
        return None
    for i in range(len(y)):
        if size[i] > 0:
            y[i] = size[i]
    return NodeShape(y, x.dtype)


_register_shape_inferer("crop_nd", infer_crop_nd)


def conv2d_forward(x, padding, dilation, kernel, stride):
    return math.floor((x + padding - (dilation * (kernel - 1) + 1)) / stride + 1)


def infer_conv2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2
    x = inputs[0]
    w = node.inputs[1]
    if w.op != Node.Const:
        return None
    w = w.get("value")
    w = tensor.from_any(w)

    padding = node.get("padding")
    type = str(node.get("format"))
    stride = node.get("stride")
    dilation = node.get("dilation")
    kernel = w.shape[-2:]

    padding = numpy.asarray(padding)

    y = list(x.shape)
    plant = ()
    if type == "NCHW":
        plant = (2, 3)
    elif type == "NHWC":
        plant = (1, 2)
    else:
        return None

    for p in range(len(plant)):
        i = plant[p]
        y[i] = conv2d_forward(x.shape[i], padding[i, 0] + padding[i, 1], dilation[i], kernel[p], stride[i])

    return NodeShape(y, x.dtype)


_register_shape_inferer("conv2d", infer_conv2d)


def infer_copy_0(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) > 0

    x = inputs[0]
    return NodeShape(x.shape, x.dtype)


_register_shape_inferer("add_bias", infer_copy_0)
_register_shape_inferer("relu", infer_copy_0)
_register_shape_inferer("softmax", infer_copy_0)


def pooling2d_forward(x, padding, kernel, stride):
    return math.ceil((x + padding - kernel) / float(stride) + 1)


def infer_pooling2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1
    x = inputs[0]

    padding = node.get("padding")
    type = str(node.get("format"))
    stride = node.get("stride")
    kernel = node.get("ksize")

    padding = numpy.asarray(padding)

    y = list(x.shape)
    plant = ()
    if type == "NCHW":
        plant = (2, 3)
    elif type == "NHWC":
        plant = (1, 2)
    else:
        return None

    for p in range(len(plant)):
        i = plant[p]
        y[i] = pooling2d_forward(x.shape[i], padding[i, 0] + padding[i, 1], kernel[i], stride[i])

    return NodeShape(y, x.dtype)


_register_shape_inferer("pooling2d", infer_pooling2d)


def infer_flatten(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]

    if len(x.shape) == 0:
        return NodeShape((1, 1), x.dtype)
    elif len(x.shape) == 1:
        return NodeShape((x.shape[0], 1), x.dtype)

    y = (x.shape[0], numpy.prod(x.shape[1:]))

    return NodeShape(y, x.dtype)


_register_shape_inferer("flatten", infer_flatten)


def infer_reshape(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    y = list(node.get("shape"))

    return NodeShape(y, x.dtype)


_register_shape_inferer("_reshape", infer_reshape)


def infer_inner_prod(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    x = inputs[0]
    w = node.inputs[1]
    if w.op != Node.Const:
        return None
    w = w.get("value")
    w = tensor.from_any(w)

    transpose = node.try_get("transpose", False)

    if len(x.shape) == 0:
        return None

    a = x.shape
    a = (a[0], numpy.prod(a[1:]))

    b = w.shape

    y = ()
    if transpose:
        y = (a[0], b[0])
    else:
        y = (a[0], b[1])

    return NodeShape(y, x.dtype)


_register_shape_inferer("inner_prod", infer_inner_prod)


if __name__ == "__main__":
    a = menu.param("a", [3], FLOAT32)
    b = menu.param("b", [3], FLOAT32)
    c = menu.op("c", "add", [a, b])
    d = menu.op("d", "add", [a, c])

    print(infer(d))