#!python
# coding: UTF-8
"""
author: kier
"""

import tensorstack as ts
import numpy
import copy

class Name(object):
    class Layer(object):
        nchw2nhwc = "nchw2nhwc"
        nhwc2nchw = "nhwc2nchw"


def nchw2nhwc(x, name=None):
    if name is None:
        name = Name.Layer.nchw2nhwc
    return ts.menu.op(name, Name.Layer.nchw2nhwc, inputs=[x,])


def nhwc2nchw(x, name=None):
    if name is None:
        name = Name.Layer.nhwc2nchw
    return ts.menu.op(name, Name.Layer.nhwc2nchw, inputs=[x, ])


def map_nchw2nchw_to_nchw(x):
    # type: (ts.Node) -> ts.Node
    return x.inputs[0]


def map_copy_to_nchw(x):
    # type: (ts.Node) -> ts.Node
    chw_input = [nhwc2nchw(i, name=i.name + "_nchw") for i in x.inputs]
    x = copy.copy(x)
    ts.Node.Link(x, chw_input)
    x.name = x.name + "_nchw"
    return x


def map_fused_batch_norm_to_nchw(x):
    # type: (ts.Node) -> ts.Node
    assert x.op == "fused_batch_norm"
    chw_input = list(x.inputs)
    chw_input[0] = nhwc2nchw(chw_input[0], name=chw_input[0].name + "_nchw")
    x = copy.copy(x)
    ts.Node.Link(x, chw_input)
    x.name = x.name + "_nchw"

    if x.get("dim") != 3:
        return None

    x.set("dim", 1)

    return x


def map_reduce_mean(x):
    # type: (ts.Node) -> ts.Node
    assert x.op == "reduce_mean"
    keep_dims = x.try_get("keep_dims", True)
    if not keep_dims:
        return None
    dims = x.get("dims")
    for dim in dims:
        if dim > 3 or dim < 0:
            return None

    chw_input = list(x.inputs)
    chw_input[0] = nhwc2nchw(chw_input[0], name=chw_input[0].name + "_nchw")
    x = copy.copy(x)
    ts.Node.Link(x, chw_input)
    x.name = x.name + "_nchw"

    map_dims_nhwc2nchw = [0, 2, 3, 1]
    dims = [map_dims_nhwc2nchw[i] for i in dims]
    x.set("dims", dims)

    return x


def map_concat(x):
    # type: (ts.Node) -> ts.Node
    assert x.op == "concat"

    dim = x.get("dim")
    if dim < 0 or dim > 3:
        return None

    chw_inputs = list(x.inputs)
    chw_inputs = [nhwc2nchw(chw_input, name=chw_input.name + "_nchw") for chw_input in chw_inputs]
    x = copy.copy(x)
    ts.Node.Link(x, chw_inputs)
    x.name = x.name + "_nchw"

    map_dims_nhwc2nchw = [0, 2, 3, 1]
    dim = map_dims_nhwc2nchw[dim]
    x.set("dim", dim)

    return x


def map_reduce_sum(x):
    # type: (ts.Node) -> ts.Node
    assert x.op == "reduce_sum"
    keep_dims = x.try_get("keep_dims", True)
    if not keep_dims:
        return None
    dims = x.get("dims")
    for dim in dims:
        if dim > 3 or dim < 0:
            return None

    chw_input = list(x.inputs)
    chw_input[0] = nhwc2nchw(chw_input[0], name=chw_input[0].name + "_nchw")
    x = copy.copy(x)
    ts.Node.Link(x, chw_input)
    x.name = x.name + "_nchw"

    map_dims_nhwc2nchw = [0, 2, 3, 1]
    dims = [map_dims_nhwc2nchw[i] for i in dims]
    x.set("dims", dims)

    return x


def map_transpose(x):
    # type: (ts.Node) -> ts.Node
    assert x.op == "_transpose"

    permute = x.get("permute")
    permute = tuple(permute)

    if permute == (0, 3, 1, 2):
        return None

    if permute == (0, 2, 3, 1):
        return x.inputs[0]

    return None


supported_map = {
    Name.Layer.nchw2nhwc: map_nchw2nchw_to_nchw,
    "_copy": map_copy_to_nchw,
    "add": map_copy_to_nchw,
    "sub": map_copy_to_nchw,
    "mul": map_copy_to_nchw,
    "div": map_copy_to_nchw,
    "relu": map_copy_to_nchw,
    "relu_max": map_copy_to_nchw,
    "rsqrt": map_copy_to_nchw,
    "maximum": map_copy_to_nchw,
    "square": map_copy_to_nchw,
    "fused_batch_norm": map_fused_batch_norm_to_nchw,
    "reduce_mean": map_reduce_mean,
    "concat": map_concat,
    "reduce_sum": map_reduce_sum,
    "_transpose": map_transpose,
}

unsupported_set = {
    Name.Layer.nhwc2nchw,
    ts.Node.Parameter,
    ts.Node.Const,
    "_resize2d",
    "_transpose",
}


def try_to_nchw(x, ready_nchw=None):
    # type: (ts.Node, dict) -> Union[ts.Node, None]
    """
    :param x:
    :param kwargs:  may have ready, dict for parsed nodes
    :return:
    """
    if ready_nchw is None:
        ready_nchw = {}

    if x in ready_nchw:
        return ready_nchw[x]

    op = x.op
    if op in supported_map:
        ready_x = supported_map[op](x)
        ready_nchw[x] = ready_x
        return ready_x
    elif op in unsupported_set:
        return None
    else:
        raise NotImplementedError("{} was not marked in supported or unsupported".format(op))


def zipnode(x, ready_zipped=None, ready_nchw=None):
    # type: (ts.Node) -> ts.Node
    """
    :param x:
    :param kwargs: may have ready, dict for parsed nodes
    :return:
    """
    if ready_zipped is None:
        ready_zipped = {}

    if ready_nchw is None:
        ready_nchw = {}

    if x in ready_zipped:
        return ready_zipped[x]

    # First transform.
    try_nchw_inputs = []
    for input in x.inputs:
        assert isinstance(input, ts.Node)
        if input.op == Name.Layer.nhwc2nchw:
            tmp = try_to_nchw(input.inputs[0], ready_nchw=ready_nchw)
            if tmp is not None:
                input = tmp
        try_nchw_inputs.append(input)
    # Then traverse.
    zipped_inputs = []
    for input in try_nchw_inputs:
        zipped_inputs.append(zipnode(input, ready_zipped=ready_zipped, ready_nchw=ready_nchw))
    ts.Node.Link(x, zipped_inputs)

    if x.op == Name.Layer.nhwc2nchw:
        x.op = ts.zoo.Name.Layer.transpose
        x.params[ts.zoo.Name.permute] = numpy.asarray([0, 3, 1, 2], dtype=numpy.int32)
    elif x.op == Name.Layer.nchw2nhwc:
        x.op = ts.zoo.Name.Layer.transpose
        x.params[ts.zoo.Name.permute] = numpy.asarray([0, 2, 3, 1], dtype=numpy.int32)

    ready_zipped[x] = x

    return x


def warp_node(x, ready=None):
    # type: (ts.Node) -> ts.Node
    """
    :param x:
    :param kwargs: may have ready, dict for parsed nodes
    :return:
    """
    if ready is None:
        ready = set()

    if x in ready:
        return x

    for input in x.inputs:
        warp_node(input, ready)

    if x.op == Name.Layer.nhwc2nchw:
        x.op = ts.zoo.Name.Layer.transpose
        x.params[ts.zoo.Name.permute] = numpy.asarray([0, 3, 1, 2], dtype=numpy.int32)
    elif x.op == Name.Layer.nchw2nhwc:
        x.op = ts.zoo.Name.Layer.transpose
        x.params[ts.zoo.Name.permute] = numpy.asarray([0, 2, 3, 1], dtype=numpy.int32)

    ready.add(x)

    return x


def plot_graph(node, plot=None):
    if plot is None:
        plot = set()

    if not isinstance(node, (tuple, list)):
        node = [node,]

    for x in node:
        assert isinstance(x, ts.Node)
        if x in plot:
            continue
        plot_graph(x.inputs, plot)

    for x in node:
        assert isinstance(x, ts.Node)
        if x in plot:
            continue
        if x.op == ts.Node.Const:
            data = x.get("value")
            data = numpy.asarray(data)
            print("{}: {} = {}".format(x.op, x.name, data.shape))
        else:
            print("{}: {} -> {}".format(x.op, [i.name for i in x.inputs], x.name))
        plot.add(x)


if __name__ == '__main__':
    def inner_layer(name, x):
        x = nhwc2nchw(x)
        x = ts.zoo.sigmoid(name, x)
        x = nchw2nhwc(x)
        return x

    input = ts.menu.param("input")

    x = input
    for i in range(10):
        x = inner_layer("layer{}".format(i), x)

    output = ts.zoo.sigmoid("output", x)

    plot_graph(output)

    output = zipnode(output)

    print("----------------------------------------")

    plot_graph(output)

    print("----------------------------------------")

    input = ts.menu.param("input")

    x = ts.zoo.copy("input_bn", input)

    x = inner_layer("input_conv", x)

    left = inner_layer("left_conv", x)
    left = ts.zoo.copy("left_bn", left)
    right = x

    y = ts.zoo.add("y", left, right)

    y = inner_layer("act", y)

    output = y

    print("========================================")

    plot_graph(output)

    output = zipnode(output)

    print("----------------------------------------")

    plot_graph(output)

