#!/usr/bin/env python

"""
:author Kier
"""

from typing import Union, List, Tuple, Dict

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
        elif isinstance(obj, numpy.ndarray):
            shape = list(obj)
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
        shape = list(self.shape)
        shape = [str(i) if i > 0 else '?' for i in shape]
        return "{}:[{}]".format(ts_dtype.dtype_str(self.dtype), ", ".join(shape))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.shape)


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


def _valid_dims(shape, *dims):
    # type: (Union[NodeShape, List[int], Tuple[int]], Union[None, List[int]]) -> bool
    if len(dims) == 1 and dims[0] is None:
        dims = list(range(len(shape)))
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
        dims = dims[0]
    if len(dims) == 0:
        dims = list(range(len(shape)))
    if isinstance(shape, NodeShape):
        shape = shape.shape
    for dim in dims:
        if shape[dim] <= 0:
            return False
    return True


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

    # sys.stderr.write("Infered shape of {}:{} = {}\n".format(node.op, node.name, shape))

    node.shape = shape[0].shape
    node.dtype = shape[0].dtype
    cache[node] = shape
    return shape


def _infer_dim(a, b):
    if a <= 0:
        if b == 1:
            return -1
        else:
            return b
    if a == 1:
        return b
    if b <= 0:
        return a
    if b == 1:
        return a
    if a == b:
        return a
    raise Exception("Can not reduce {} with {}".format(a, b))


def infer_eltwise(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2
    import numpy
    a = inputs[0]
    b = inputs[1]
    if _valid_dims(a) and _valid_dims(b):
        c = numpy.zeros(a.shape) + numpy.zeros(b.shape)
        return NodeShape(c.shape, c.dtype)
    dtype = a.dtype
    a = a.shape
    b = b.shape
    if len(a) < len(b):
        a, b = b, a
    b = list(b)
    while len(a) > len(b):
        b.insert(0, 1)
    c = [-1] * len(a)
    try:
        for i in range(len(c)):
            c[i] = _infer_dim(a[i], b[i])
        return NodeShape(c, dtype)
    except:
        sys.stderr.write("Failed infer shape of {}:{} with {}, {}\n".format(node.op, node.name, inputs[0], inputs[1]))
        return None


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
    channel = 0
    if type == "NCHW":
        plant = (2, 3)
        channel = 1
    elif type == "NHWC":
        plant = (1, 2)
        channel = 3
    else:
        return None

    for p in range(len(plant)):
        i = plant[p]
        if x.shape[i] < 0:
            y[i] = -1
            continue
        y[i] = conv2d_forward(x.shape[i], padding[i, 0] + padding[i, 1], dilation[i], kernel[p], stride[i])
    y[channel] = w.shape[0]

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
        if x.shape[i] < 0:
            y[i] = -1
            continue
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

    for i in range(len(y)):
        if y[i] == 0:
            y[i] = x.shape[i]

    if _valid_dims(x):
        tmp = numpy.zeros(x.shape).reshape(y)
        return NodeShape(tmp.shape, x.dtype)

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

    # print("{}x{}".format(a, b))

    y = ()
    if transpose:
        y = (a[0], b[0])
    else:
        y = (a[0], b[1])

    return NodeShape(y, x.dtype)


_register_shape_inferer("inner_prod", infer_inner_prod)


_register_shape_inferer("batch_norm", infer_copy_0)
_register_shape_inferer("batch_scale", infer_copy_0)


def infer_concat(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) > 1

    dim = int(node.get("dim"))

    shape = list(inputs[0].shape)
    for i in range(1, len(inputs)):
        shape[dim] += inputs[i].shape[dim]

    return NodeShape(shape, inputs[0].dtype)


_register_shape_inferer("concat", infer_concat)


def infer_global_pooling2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1
    x = inputs[0]

    type = str(node.get("format"))

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
        y[i] = 1

    return NodeShape(y, x.dtype)


_register_shape_inferer("global_pooling2d", infer_global_pooling2d)


_register_shape_inferer("sigmoid", infer_copy_0)


def infer_dims(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1
    return NodeShape([], ts_dtype.INT32)


_register_shape_inferer("_dims", infer_dims)


def infer_expand(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    x = inputs[0]
    dims = node.inputs[1]

    assert isinstance(dims, Node)

    if dims.op == Node.Const:
        dims = int(dims.get("value"))
    elif dims.op == "_dims":
        y = dims.inputs[0]
        assert isinstance(y, Node)
        if not has_infered(y):
            return None
        dims = len(y.shape)
    else:
        return None

    front = node.try_get("front", dims)
    end = node.try_get("end", dims)
    inverse = node.try_get("inverse", False)
    y = list(x.shape)

    if not inverse:
        while len(y) < dims and front > 0:
            y.insert(0, 1)
        while len(y) < dims and end > 0:
            y.append(1)
    else:
        while len(y) < dims and end > 0:
            y.append(1)
        while len(y) < dims and front > 0:
            y.insert(0, 1)

    return NodeShape(y, x.dtype)


_register_shape_inferer("_expand", infer_expand)


def infer_limit(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1
    x = inputs[0]
    shape = list(node.get("shape"))

    while len(shape) < len(x.shape):
        shape.insert(0, -1)

    y = copy.copy(shape)
    for i in range(len(y)):
        if y[i] < 0:
            y[i] = x.shape[i]
        elif x.shape[i] < 0:
            y[i] = shape[i]
        else:
            y[i] = min(y[i], x.shape[i])

    return NodeShape(y, x.dtype)


_register_shape_inferer("_limit", infer_limit)


_register_shape_inferer("_copy", infer_copy_0)
_register_shape_inferer("prelu", infer_copy_0)


def transpose_conv2d_forward(x, padding, dilation, kernel, stride):
    return (x - 1) * stride + (dilation * (kernel - 1) + 1) - padding


def infer_transpose_conv2d(node, inputs):
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
    channel = 0
    if type == "NCHW":
        plant = (2, 3)
        channel = 1
    elif type == "NHWC":
        plant = (1, 2)
        channel = 3
    else:
        return None

    for p in range(len(plant)):
        i = plant[p]
        if x.shape[i] < 0:
            y[i] = -1
            continue
        y[i] = transpose_conv2d_forward(x.shape[i], padding[i, 0] + padding[i, 1], dilation[i], kernel[p], stride[i])
    y[channel] = w.shape[1]

    return NodeShape(y, x.dtype)


_register_shape_inferer("transpose_conv2d", infer_transpose_conv2d)


if __name__ == "__main__":
    a = menu.param("a", [3], FLOAT32)
    b = menu.param("b", [3], FLOAT32)
    c = menu.op("c", "add", [a, b])
    d = menu.op("d", "add", [a, c])

    print(infer(d))