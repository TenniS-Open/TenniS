#!/usr/bin/env python

"""
:author Kier
"""

from typing import Union, List, Tuple, Dict, Set

from .node import Node
from . import dtype as ts_dtype
from .dtype import VOID, FLOAT32
from . import tensor
from . import menu
from . import _inferer_

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


def infer(node, endpoints=None, cache=None):
    # type: (Union[Node, List[Node]], List[Node], Dict[Node, List[NodeShape]]) -> Union[List[NodeShape], List[List[NodeShape]]]
    """
    For more information, the #padding attr will be set in conv2d_v2 and pooling2d_v2,
        #value will be set in the node can be compute in static.
    :param node: node want to be infer
    :param endpoints: end points about graph, so not infer
    :param cache: infered nodes
    :return: list of NodeShape
    """
    if cache is None:
        cache = {}

    if isinstance(node, (list, tuple)):
        return [infer(n, cache=cache) for n in node]

    if node in cache:
        return cache[node]

    if endpoints is not None and node in endpoints:
        if not has_infered(node):
            raise Exception("Node {}:{} must be set shape and dtype".format(node.op, node.name))
        return [NodeShape(node)]

    inputs = [infer(i, cache=cache) for i in node.inputs]
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

    inputs = [i[0] if len(i) == 1 else i for i in inputs]
    shape = _shape_inferer[node.op](node, inputs)

    if isinstance(shape, NodeShape):
        shape = [shape]

    if shape is None or len(shape) == 0:
        sys.stderr.write("Failed infer shape of {}:{}\n".format(node.op, node.name))
        node.dtype = VOID
        node.shape = [-1]
        cache[node] = []
        return []

    # if node.op != Node.Const:
    #     sys.stderr.write("Infered shape of {}:{} = {}\n".format(node.op, node.name, shape))

    node.shape = shape[0].shape
    node.dtype = shape[0].dtype
    cache[node] = shape
    return shape


def _infer_value(node, value=None):
    # type: (Node, object) -> Union[None, numpy.ndarray]
    if node.op == Node.Const:
        return node.get("value")
    return node.try_get("#value", value)


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


class EltwiseInferer(object):
    def __init__(self, func=None):
        self.__func = func

    def __call__(self, node, inputs):
        # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
        assert len(inputs) == 2
        import numpy
        a = inputs[0]
        b = inputs[1]
        dtype = a.dtype
        if _valid_dims(a) and _valid_dims(b):
            c = numpy.zeros(a.shape) + numpy.zeros(b.shape)
            c = c.shape
            # return NodeShape(c.shape, a.dtype)
        else:
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
            except:
                sys.stderr.write("Failed infer shape of {}:{} with {}, {}\n".format(node.op, node.name, inputs[0], inputs[1]))
                return None

        lhs = _infer_value(node.inputs[0])
        rhs = _infer_value(node.inputs[1])

        if self.__func is not None and lhs is not None and rhs is not None:
            node.set("#value", self.__func(lhs, rhs), dtype=dtype)

        return NodeShape(c, dtype)


_register_shape_inferer("add", EltwiseInferer(lambda a, b: numpy.asarray(a) + numpy.asarray(b)))
_register_shape_inferer("sub", EltwiseInferer(lambda a, b: numpy.asarray(a) - numpy.asarray(b)))
_register_shape_inferer("mul", EltwiseInferer(lambda a, b: numpy.asarray(a) * numpy.asarray(b)))
_register_shape_inferer("div", EltwiseInferer(lambda a, b: numpy.asarray(a) / numpy.asarray(b)))


def infer_to_float(node, inputs):
    # type: (Node, List[NodeShape]) -> NodeShape
    assert len(inputs) == 1
    x = inputs[0]

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", a, dtype=numpy.float32)

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
    x_shape = list(x.shape)
    while len(x_shape) < len(permute):
        x_shape.insert(0, 1)
    y = [0] * len(x_shape)
    for i in range(len(y)):
        y[i] = x_shape[permute[i]]

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", numpy.transpose(numpy.reshape(a, x_shape), axes=permute))

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
    fmt = str(node.get("format"))
    stride = node.get("stride")
    dilation = node.get("dilation")
    kernel = w.shape[-2:]

    padding = numpy.asarray(padding)

    y = list(x.shape)
    plant = ()
    channel = 0
    if fmt == "NCHW":
        plant = (2, 3)
        channel = 1
    elif fmt == "NHWC":
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
    fmt = str(node.get("format"))
    stride = node.get("stride")
    kernel = node.get("ksize")

    padding = numpy.asarray(padding)

    y = list(x.shape)
    plant = ()
    if fmt == "NCHW":
        plant = (2, 3)
    elif fmt == "NHWC":
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

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", numpy.reshape(a, (a.shape[0], -1)))

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
        y = tmp.shape

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", numpy.reshape(a, y))

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
    assert len(inputs) >= 1

    dim = int(node.get("dim"))

    shape = list(inputs[0].shape)
    for i in range(1, len(inputs)):
        shape[dim] += inputs[i].shape[dim]

    a = [_infer_value(i) for i in node.inputs]
    ready = True
    for i in a:
        if i is None:
            ready = False
            break
    if ready:
        a = numpy.concatenate(a, axis=dim)
        node.set("#value", a)

    return NodeShape(shape, inputs[0].dtype)


_register_shape_inferer("concat", infer_concat)


def infer_global_pooling2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1
    x = inputs[0]

    fmt = str(node.get("format"))

    y = list(x.shape)
    plant = ()
    if fmt == "NCHW":
        plant = (2, 3)
    elif fmt == "NHWC":
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

    node.set("#value", len(inputs[0].shape), numpy.int32)

    return NodeShape([], ts_dtype.INT32)


_register_shape_inferer("_dims", infer_dims)


def infer_expand(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    x = inputs[0]
    dims = node.inputs[1]

    assert isinstance(dims, Node)

    dims = _infer_value(dims)
    if dims is None:
        return
    dims = int(dims)

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

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", numpy.reshape(a, y), x.dtype)

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


def infer_copy(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) > 0

    x = inputs[0]

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", a)

    return NodeShape(x.shape, x.dtype)


_register_shape_inferer("_copy", infer_copy)

_register_shape_inferer("prelu", infer_copy_0)


def conv2d_backward(y, padding, dilation, kernel, stride):
    return (y - 1) * stride + (dilation * (kernel - 1) + 1) - padding


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
    fmt = str(node.get("format"))
    stride = node.get("stride")
    dilation = node.get("dilation")
    kernel = w.shape[-2:]

    padding = numpy.asarray(padding)

    y = list(x.shape)
    plant = ()
    channel = 0
    if fmt == "NCHW":
        plant = (2, 3)
        channel = 1
    elif fmt == "NHWC":
        plant = (1, 2)
        channel = 3
    else:
        return None

    for p in range(len(plant)):
        i = plant[p]
        if x.shape[i] < 0:
            y[i] = -1
            continue
        y[i] = conv2d_backward(x.shape[i], padding[i, 0] + padding[i, 1], dilation[i], kernel[p], stride[i])
    y[channel] = w.shape[1]

    return NodeShape(y, x.dtype)


_register_shape_inferer("transpose_conv2d", infer_transpose_conv2d)


def infer_shape_index_patch(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2
    x = inputs[0]
    pos = inputs[1]

    number, channels, height, width = x.shape
    number2, landmark = pos.shape[:2]

    assert number == number2

    origin_patch = list(node.get("origin_patch"))
    origin = list(node.get("origin"))

    x_patch_h = int(origin_patch[0] * height / origin[0] + 0.5)
    x_patch_w = int(origin_patch[1] * width / origin[1] + 0.5)

    return NodeShape((number, channels, x_patch_h, landmark // 2, x_patch_w), x.dtype)


_register_shape_inferer("shape_index_patch", infer_shape_index_patch)


def infer_cast(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1
    x = inputs[0]

    dtype = int(node.get("dtype"))

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", a, dtype=dtype)

    return NodeShape(x.shape, dtype)


_register_shape_inferer("_cast", infer_cast)


def infer_depthwise_conv2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2
    x = inputs[0]

    w = node.inputs[1]
    if w.op != Node.Const:
        return None
    w = w.get("value")
    w = tensor.from_any(w)

    padding = node.get("padding")
    fmt = str(node.get("format"))
    stride = node.get("stride")
    dilation = node.get("dilation")
    kernel = w.shape[-2:]

    padding = numpy.asarray(padding)

    y = list(x.shape)
    plant = ()
    channel = 0
    if fmt == "NCHW":
        plant = (2, 3)
        channel = 1
    elif fmt == "NHWC":
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
    y[channel] = w.shape[0] * x.shape[channel]

    return NodeShape(y, x.dtype)


_register_shape_inferer("depthwise_conv2d", infer_depthwise_conv2d)


def infer_dynamic_padding(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    return NodeShape([4, 2], ts_dtype.INT32)


_register_shape_inferer("_onnx_pooling2d_padding", infer_dynamic_padding)
_register_shape_inferer("_dragon_pooling2d_padding", infer_dynamic_padding)
_register_shape_inferer("_mx_pooling2d_padding", infer_dynamic_padding)
_register_shape_inferer("_tf_conv2d_padding", infer_dynamic_padding)
_register_shape_inferer("_tf_pooling2d_padding", infer_dynamic_padding)
_register_shape_inferer("_dragon_conv2d_padding", infer_dynamic_padding)


_dynamic_padding_factory = {
    # pooling2d
    "_onnx_pooling2d_padding": _inferer_.onnx_pooling2d_padding,
    "_dragon_pooling2d_padding": _inferer_.dragon_pooling2d_padding,
    "_mx_pooling2d_padding": _inferer_.mx_pooling2d_padding,
    "_tf_pooling2d_padding": _inferer_.tf_pooling2d_padding,
    # conv2d
    "_tf_conv2d_padding": _inferer_.tf_conv2d_padding,
    "_dragon_conv2d_padding": _inferer_.dragon_conv2d_padding,
}


def _register_dynamic_padding(op, padding):
    _dynamic_padding_factory[op] = padding


def infer_pooling2d_v2(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 4
    x = inputs[0]

    # == calculate dynamic padding
    padding_node = node.inputs[1]
    assert isinstance(padding_node, Node)
    padding_op = padding_node.op
    if padding_op not in _dynamic_padding_factory:
        return None
    padding = _dynamic_padding_factory[padding_op](padding_node)
    if padding is None:
        return None
    # == end
    padding = numpy.asarray(padding)

    # == get stride
    ksize_node = node.inputs[2]
    if ksize_node.op != Node.Const:
        return None

    # == get ksize
    stride_node = node.inputs[3]
    if stride_node.op != Node.Const:
        return None

    stride = stride_node.get("value")
    kernel = ksize_node.get("value")

    fmt = str(node.get("format"))
    y = list(x.shape)
    plant = ()
    if fmt == "NCHW":
        plant = (2, 3)
    elif fmt == "NHWC":
        plant = (1, 2)
    else:
        return None

    for p in range(len(plant)):
        i = plant[p]
        if x.shape[i] < 0:
            y[i] = -1
            continue
        y[i] = pooling2d_forward(x.shape[i], padding[i, 0] + padding[i, 1], kernel[i], stride[i])

    node.set("#padding", padding, dtype=numpy.int32)

    return NodeShape(y, x.dtype)


_register_shape_inferer("pooling2d_v2", infer_pooling2d_v2)


def infer_conv2d_v2(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 3
    x = inputs[0]

    # == calculate dynamic padding
    padding_node = node.inputs[1]
    assert isinstance(padding_node, Node)
    padding_op = padding_node.op
    if padding_op not in _dynamic_padding_factory:
        return None
    padding = _dynamic_padding_factory[padding_op](padding_node)
    if padding is None:
        return None
    # == end
    padding = numpy.asarray(padding)

    w = node.inputs[2]
    if w.op != Node.Const:
        return None
    w = w.get("value")
    w = tensor.from_any(w)

    stride = node.get("stride")
    dilation = node.get("dilation")
    kernel = w.shape[-2:]

    fmt = str(node.get("format"))
    y = list(x.shape)
    plant = ()
    channel = 0
    if fmt == "NCHW":
        plant = (2, 3)
        channel = 1
    elif fmt == "NHWC":
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

    node.set("#padding", padding, dtype=numpy.int32)

    return NodeShape(y, x.dtype)


_register_shape_inferer("conv2d_v2", infer_conv2d_v2)


def infer_depthwise_conv2d_v2(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 3
    x = inputs[0]

    # == calculate dynamic padding
    padding_node = node.inputs[1]
    assert isinstance(padding_node, Node)
    padding_op = padding_node.op
    if padding_op not in _dynamic_padding_factory:
        return None
    padding = _dynamic_padding_factory[padding_op](padding_node)
    if padding is None:
        return None
    # == end
    padding = numpy.asarray(padding)

    w = node.inputs[2]
    if w.op != Node.Const:
        return None
    w = w.get("value")
    w = tensor.from_any(w)

    stride = node.get("stride")
    dilation = node.get("dilation")
    kernel = w.shape[-2:]

    fmt = str(node.get("format"))
    y = list(x.shape)
    plant = ()
    channel = 0
    if fmt == "NCHW":
        plant = (2, 3)
        channel = 1
    elif fmt == "NHWC":
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
    y[channel] = w.shape[0] * x.shape[channel]

    node.set("#padding", padding, dtype=numpy.int32)

    return NodeShape(y, x.dtype)


_register_shape_inferer("depthwise_conv2d_v2", infer_depthwise_conv2d_v2)


def infer_gemm(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 3

    A = inputs[0]
    B = inputs[1]

    # print("{}x{}".format(A.shape, B.shape))

    transA = node.get("transA")
    transB = node.get("transB")

    M = A.shape[0] if not transA else A.shape[1]
    N = B.shape[1] if not transB else B.shape[0]

    return NodeShape((M, N), A.dtype)


_register_shape_inferer("gemm", infer_gemm)


_register_shape_inferer("abs", infer_copy_0)


def infer_shape(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]

    node.set("#value", x.shape, dtype=numpy.int32)

    return NodeShape((len(x.shape),), ts_dtype.INT32)


_register_shape_inferer("_shape", infer_shape)


def infer_gather(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    axis = int(node.get("axis"))
    x = inputs[0]
    indices_shape = inputs[1].shape

    if axis < 0:
        axis = len(x) + axis

    y = list(x.shape)
    del y[axis]

    anchor = axis
    for i in indices_shape:
        y.insert(anchor, i)
        anchor += 1

    # == infer value
    a = _infer_value(node.inputs[0])
    i = _infer_value(node.inputs[1])
    if a is not None and i is not None:
        node.set("#value", numpy.take(a, i, axis=axis))

    return NodeShape(y, x.dtype)


_register_shape_inferer("gather", infer_gather)


def infer_field(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    offset = node.get("offset")

    x = inputs[0]
    if isinstance(x, NodeShape):
        x = [x]

    if offset >= len(x):
        return None

    y = x[offset]

    return NodeShape(y.shape, y.dtype)


_register_shape_inferer("_field", infer_field)


def infer_reshape_v2(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    x = inputs[0]
    shape = node.inputs[1]

    shape = _infer_value(shape)
    if shape is None:
        return None

    y = list(shape)

    for i in range(len(y)):
        if y[i] == 0:
            y[i] = x.shape[i]

    if _valid_dims(x):
        tmp = numpy.zeros(x.shape).reshape(y)
        y = tmp.shape

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", numpy.reshape(a, y))

    return NodeShape(y, x.dtype)


_register_shape_inferer("_reshape_v2", infer_reshape_v2)


def infer_stack(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) >= 1

    axis = int(node.get("axis"))

    if axis < 0:
        axis += len(inputs[0].shape) + 1

    shape = list(inputs[0].shape)
    shape.insert(axis, len(inputs))

    a = [_infer_value(i) for i in node.inputs]
    ready = True
    for i in a:
        if i is None:
            ready = False
            break
    if ready:
        a = numpy.stack(a, axis=axis)
        node.set("#value", a)

    return NodeShape(shape, inputs[0].dtype)


_register_shape_inferer("stack", infer_stack)


_register_shape_inferer("fused_batch_norm", infer_copy_0)
_register_shape_inferer("l2_norm", infer_copy_0)


def infer_dimshuffle(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    dim = int(node.get("dim"))
    x = inputs[0]
    shuffle = node.get("shuffle")

    shuffle = numpy.asarray(shuffle).reshape([-1])

    if dim < 0:
        dim = len(x) + dim

    y = list(x.shape)
    y[dim] = len(shuffle)

    # == infer value
    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", numpy.take(a, shuffle, axis=dim))

    return NodeShape(y, x.dtype)


_register_shape_inferer("_dimshuffle", infer_dimshuffle)


def infer_squeeze(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    axes = node.try_get("axes", None)

    y = list(x.shape)

    if axes is None:
        tmp = []
        for i in y:
            if i != 1:
                tmp.append(i)
        y = tmp
    else:
        for axis in axes[::-1]:
            # if y[axis] != 1:
            #     return None
            y.pop(axis)

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", numpy.reshape(a, y), x.dtype)

    return NodeShape(y, x.dtype)


_register_shape_inferer("squeeze", infer_squeeze)


def infer_sample2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1
    x = inputs[0]

    scale = float(node.get("scale"))
    dim = int(node.try_get("dim", -2))

    if dim < 0:
        dim += len(x)

    if dim < 0 or dim + 1 >= len(x):
        return None

    y = list(x.shape)

    if y[dim] > 0:
        y[dim] = int(y[dim] * scale)
    if y[dim + 1] > 0:
        y[dim + 1] = int(y[dim + 1] * scale)

    return NodeShape(y, x.dtype)


_register_shape_inferer("sample2d", infer_sample2d)


def infer_proposal(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) >= 3
    x = inputs[0]

    min_level = int(node.try_get("min_level", 2))
    max_level = int(node.try_get("max_level", 5))
    post_nms_top_n = int(node.try_get("post_nms_top_n", 300))

    num_images = inputs[0].shape[0]

    output_size = max_level - min_level + 1

    output = []
    for i in range(output_size):
        proto = NodeShape((num_images * post_nms_top_n if num_images > 0 else -1, 5), inputs[-3].dtype)
        output.append(proto)

    return output


_register_shape_inferer("proposal", infer_proposal)


def infer_roi_align(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    pool_h = int(node.get("pool_h"))
    pool_w = int(node.get("pool_w"))

    return NodeShape((inputs[1].shape[0], inputs[0].shape[1], pool_h, pool_w), inputs[0].dtype)


_register_shape_inferer("roi_align", infer_roi_align)


_register_shape_inferer("relu_max", infer_copy_0)


def infer_reduce_mean(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    dims = list(numpy.asarray(node.get("dims")).reshape([-1]))
    keep_dims = bool(node.try_get("keep_dims", True))

    x = inputs[0]
    y = list(x.shape)

    if keep_dims:
        for dim in dims:
            if dim < 0:
                dim += len(x)
            y[dim] = 1
    else:
        for dim in dims[::-1]:
            if dim < 0:
                dim += len(x)
            y.pop(dim)

    return NodeShape(y, x.dtype)


_register_shape_inferer("reduce_mean", infer_reduce_mean)


_register_shape_inferer("square", infer_copy_0)
_register_shape_inferer("maximum",  EltwiseInferer(lambda a, b: numpy.maximum(a, b)))
_register_shape_inferer("rsqrt", infer_copy_0)
_register_shape_inferer("exp", infer_copy_0)


def infer_reduce_sum(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    dims = list(numpy.asarray(node.get("dims")).reshape([-1]))
    keep_dims = bool(node.try_get("keep_dims", True))

    x = inputs[0]
    y = list(x.shape)

    if keep_dims:
        for dim in dims:
            if dim < 0:
                dim += len(x)
            y[dim] = 1
    else:
        for dim in dims[::-1]:
            if dim < 0:
                dim += len(x)
            y.pop(dim)

    return NodeShape(y, x.dtype)


_register_shape_inferer("reduce_sum", infer_reduce_sum)


def infer_strided_slice(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1


    begin = list(node.get("begin"))
    end = list(node.get("end"))

    if node.has("stride"):
        stride = list(node.get("stride"))
    else:
        stride = [1] * len(begin)

    begin_mask = int(node.get("begin_mask"))
    end_mask = int(node.get("end_mask"))
    ellipsis_mask = int(node.get("ellipsis_mask"))
    new_axis_mask = int(node.get("new_axis_mask"))
    shrink_axis_mask = int(node.get("shrink_axis_mask"))

    x = inputs[0]

    y, shape = _inferer_.strided_slice.infer_stride_slice(_infer_value(node.inputs[0]),
                                                          list(x.shape),
                                                          begin, end, stride,
                                                          begin_mask, end_mask,
                                                          ellipsis_mask, new_axis_mask, shrink_axis_mask)

    if shape is None:
        return None

    if y is not None:
        node.set("#value", y)

    return NodeShape(shape, x.dtype)


_register_shape_inferer("strided_slice", infer_strided_slice)


def infer_max(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    dims = list(numpy.asarray(node.get("dim")).reshape([-1]))
    keep_dims = bool(node.try_get("keep_dims", True))

    x = inputs[0]
    y = list(x.shape)

    if keep_dims:
        for dim in dims:
            if dim < 0:
                dim += len(x)
            y[dim] = 1
    else:
        for dim in dims[::-1]:
            if dim < 0:
                dim += len(x)
            y.pop(dim)

    return NodeShape(y, x.dtype)


_register_shape_inferer("max", infer_max)


def infer_topkv2(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    K = int(node.get("number"))

    if len(x.shape) == 0:
        return NodeShape(x.shape, x.dtype), NodeShape(x.shape, ts_dtype.INT32)

    K = min(x.shape[-1], K)
    y = list(x.shape)
    y[-1] = K

    return NodeShape(y, x.dtype), NodeShape(y, ts_dtype.INT32)


_register_shape_inferer("topkv2", infer_topkv2)


def non_max_suppression_v3(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    boxes = inputs[0]
    scores = inputs[1]

    max_output_size = int(node.get("max_output_size"))

    K = min(scores.shape[0], max_output_size)

    return NodeShape((K,), ts_dtype.INT32)


_register_shape_inferer("non_max_suppression_v3", non_max_suppression_v3)


def infer_argmax(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    dim = int(node.get("dim"))

    x = inputs[0]
    y = list(x.shape)

    if dim < 0:
        dim += len(x)
    y.pop(dim)

    return NodeShape(y, x.dtype)


_register_shape_inferer("argmax", infer_argmax)


def infer_unsqueeze(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    axes = numpy.asarray(node.get("axes"), dtype=numpy.int32).reshape([-1])

    y = list(x.shape)

    for axis in axes:
        if axis < 0:
            axis += len(y) + 1
        y.insert(axis, 1)

    a = _infer_value(node.inputs[0])
    if a is not None:
        node.set("#value", numpy.reshape(a, y), x.dtype)

    return NodeShape(y, x.dtype)


_register_shape_inferer("unsqueeze", infer_unsqueeze)


def infer_pad(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    x = inputs[0]
    padding = _infer_value(node.inputs[1])
    if padding is None:
        return padding
    padding = numpy.asarray(padding, dtype=numpy.int32).reshape([-1, 2])

    y = list(x.shape)

    for i in range(len(y)):
        if y[i] < 0:
            continue
        y[i] += padding[i, 0] + padding[i, 1]

    return NodeShape(y, x.dtype)


_register_shape_inferer("pad", infer_pad)


def infer_pack(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) >= 1

    return tuple(inputs)


_register_shape_inferer("_pack", infer_pack)

_register_shape_inferer("prewhiten", infer_copy_0)


def infer_batch_to_space4d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]

    crop = tuple(numpy.asarray(node.get("crop"), dtype=numpy.int32).reshape([-1]))
    block_shape = tuple(numpy.asarray(node.get("block_shape"), dtype=numpy.int32).reshape([-1]))

    block_height, block_width = block_shape
    crop_top, crop_bottom, crop_left, crop_right = crop
    input_shape = x.shape
    output_shape = [-1] * 4

    output_shape[0] = -1 if input_shape[0] < 0 else input_shape[0] // (block_height * block_width)
    output_shape[2] = -1 if input_shape[2] < 0 else input_shape[2] * block_height - crop_top - crop_bottom
    output_shape[3] = -1 if input_shape[3] < 0 else input_shape[3] * block_width - crop_left - crop_right
    output_shape[1] = -1 if input_shape[1] < 0 else input_shape[1]

    return NodeShape(output_shape, x.dtype)


_register_shape_inferer("batch_to_space4d", infer_batch_to_space4d)


def infer_space_to_batch4d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]

    padding = tuple(numpy.asarray(node.get("padding"), dtype=numpy.int32).reshape([-1]))
    block_shape = tuple(numpy.asarray(node.get("block_shape"), dtype=numpy.int32).reshape([-1]))

    block_height, block_width = block_shape
    padding_top, padding_bottom, padding_left, padding_right = padding
    input_shape = x.shape
    output_shape = [-1] * 4

    output_shape[0] = -1 if input_shape[0] < 0 else input_shape[0] * block_height * block_width
    output_shape[2] = -1 if input_shape[2] < 0 else (input_shape[2] + padding_top + padding_bottom) // block_height
    output_shape[3] = -1 if input_shape[3] < 0 else (input_shape[3] + padding_left + padding_right) // block_width
    output_shape[1] = -1 if input_shape[1] < 0 else input_shape[1]

    return NodeShape(output_shape, x.dtype)


_register_shape_inferer("space_to_batch4d", infer_space_to_batch4d)


def infer_affine_sample2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 4

    x = inputs[0]
    size = numpy.asarray(_infer_value(node.inputs[1]), dtype=numpy.int32).reshape([-1])

    dim = node.get("dim")
    if dim < 0:
        dim += len(x)

    y = list(x.shape)

    y[dim] = size[0]
    y[dim + 1] = size[1]

    return NodeShape(y, x.dtype)


_register_shape_inferer("affine_sample2d", infer_affine_sample2d)


def infer_chunk(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    chunks = int(node.get("chunks"))
    dim = int(node.get("dim"))

    dim = node.get("dim")
    if dim < 0:
        dim += len(x)

    y = list(x.shape)

    x_value = _infer_value(node.inputs[0])
    if x_value is None:
        a = numpy.zeros(x.shape)
    else:
        a = x_value

    b = numpy.split(a, chunks, dim)

    y = [NodeShape(i.shape, x.dtype) for i in b]

    if x_value is not None:
        pass

    return y


_register_shape_inferer("chunk", infer_chunk)


def infer_dcn_v2_forward(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 5
    x = inputs[0]

    w = node.inputs[1]
    if w.op != Node.Const:
        return None
    w = w.get("value")
    w = tensor.from_any(w)

    padding = node.get("padding")
    fmt = str(node.get("format"))
    stride = node.get("stride")
    dilation = node.get("dilation")
    kernel = w.shape[-2:]

    padding = numpy.asarray(padding)

    y = list(x.shape)
    plant = ()
    channel = 0
    if fmt == "NCHW":
        plant = (2, 3)
        channel = 1
    elif fmt == "NHWC":
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


_register_shape_inferer("dcn_v2_forward", infer_dcn_v2_forward)


def infer_nhwc_center_crop2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    size = numpy.asarray(node.get("size"), dtype=numpy.int32).reshape([-1])

    assert len(x) == 4

    y = list(x.shape)

    y[1], y[2] = size[1], size[0]

    return NodeShape(y, x.dtype)


_register_shape_inferer("_nhwc_center_crop2d", infer_nhwc_center_crop2d)


def infer_nhwc_scale_resize2d(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    size = numpy.asarray(node.get("size"), dtype=numpy.int32).reshape([-1])

    assert len(x) == 4
    y = list(x.shape)

    if len(size) == 1:
        h = y[1]
        w = y[2]
        if h > w:
            y[1], y[2] = size[0] * h / w, size[0]
        else:
            y[1], y[2] = size[0], size[0] * w / h
    elif len(size) == 2:
        y[1], y[2] = size[1], size[0]
    else:
        return None

    return NodeShape(y, x.dtype)


_register_shape_inferer("_nhwc_scale_resize2d", infer_nhwc_scale_resize2d)


def infer_nhwc_letterbox(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    size = numpy.asarray(node.get("size"), dtype=numpy.int32).reshape([-1])

    assert len(x) == 4
    y = list(x.shape)

    if len(size) == 1:
        y[1], y[2] = size[0], size[0]
    elif len(size) == 2:
        y[1], y[2] = size[1], size[0]
    else:
        return None

    return NodeShape(y, x.dtype)


_register_shape_inferer("_nhwc_letterbox", infer_nhwc_letterbox)


def infer_divided(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    size = numpy.asarray(node.get("size"), dtype=numpy.int32).reshape([-1])

    if len(size) > len(x):
        return None

    size = list(size)
    while len(size) < len(x):
        size.insert(0, 1)

    y = list(x.shape)

    for i in range(len(y)):
        if size[i] == 1:
            continue
        y[i] = int(math.ceil(float(y[i]) / size[i])) * size[i]

    return NodeShape(y, x.dtype)


_register_shape_inferer("divided", infer_divided)


def infer_yolo(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    if len(x) != 4:
        return None

    classes = int(node.get("classes"))
    mask = numpy.asarray(node.get("mask"), dtype=numpy.int32).reshape([-1])
    anchors = numpy.asarray(node.get("anchors"), dtype=numpy.float32).reshape([-1])
    n = len(mask)

    batch = x.shape[0]
    h = x.shape[2]
    w = x.shape[3]

    outputs = [
        NodeShape((batch, n * (classes + 4 + 1), h, w), x.dtype),
        NodeShape([], ts_dtype.INT32),
        NodeShape([len(mask)], ts_dtype.INT32),
        NodeShape([len(anchors)], ts_dtype.FLOAT32),
    ]

    return outputs


_register_shape_inferer("yolo", infer_yolo)


def infer_yolo_poster(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) >= 1

    yolo = inputs[-1]
    if isinstance(yolo, (tuple, list)):
        yolo = yolo[0]

    assert isinstance(yolo, NodeShape)

    n = yolo.shape[0]
    if n <= 0:
        n = 1

    return [NodeShape((-1, 6), FLOAT32)] * n


_register_shape_inferer("yolo_poster", infer_yolo_poster)


_register_shape_inferer("tanh", infer_copy_0)


def infer_force_gray(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    y = list(x.shape)

    y[-1] = 1

    return NodeShape(y, x.dtype)


_register_shape_inferer("force_gray", infer_force_gray)


def infer_force_color(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    y = list(x.shape)

    y[-1] = 3

    return NodeShape(y, x.dtype)


_register_shape_inferer("force_color", infer_force_color)


_register_shape_inferer("norm_image", infer_copy_0)
_register_shape_inferer("sqrt", infer_copy_0)


def infer_tile(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 1

    x = inputs[0]
    repeats = numpy.asarray(node.get("repeats"), dtype=numpy.int32).reshape([-1])

    x_value = _infer_value(node.inputs[0])
    if x_value is None:
        a = numpy.zeros(x.shape)
    else:
        a = x_value

    b = numpy.tile(a, repeats)
    y = b.shape

    if x_value is not None:
        node.set("#value", b, dtype=x.dtype)

    return NodeShape(y, x.dtype)


_register_shape_inferer("tile", infer_tile)


def infer_broadcast(node, inputs):
    # type: (Node, List[NodeShape]) -> Union[None, NodeShape]
    assert len(inputs) == 2

    x = inputs[0]
    shape = _infer_value(node.inputs[1])
    if shape is not None:
        return None
    shape = list(numpy.asarray(shape, numpy.int32).reshape([-1]))

    dtype = x.dtype

    if _valid_dims(x.shape) and _valid_dims(shape):
        c = numpy.zeros(x.shape) + numpy.zeros(shape)
        c = c.shape
    else:
        y = list(x.shape)
        if len(y) > len(shape):
            return None
        while len(y) < len(shape):
            y.insert(0, 1)
        c = [-1] * len(y)
        try:
            for i in range(len(c)):
                c[i] = _infer_dim(y[i], shape[i])
                if c[i] > 0 and shape[i] > 0 and c[i] != shape[i]:
                    return None
        except:
            return None

    lhs = _infer_value(node.inputs[0])
    rhs = None if not _valid_dims(shape) else numpy.zeros(shape)

    if lhs is not None and rhs is not None:
        node.set("#value", numpy.asarray(lhs) + numpy.asarray(rhs), dtype=dtype)

    return NodeShape(c, dtype)


_register_shape_inferer("broadcast", infer_broadcast)


if __name__ == "__main__":
    a = menu.param("a", [3], FLOAT32)
    b = menu.param("b", [3], FLOAT32)
    c = menu.op("c", "add", [a, b])
    d = menu.op("d", "add", [a, c])

    print(infer(d))