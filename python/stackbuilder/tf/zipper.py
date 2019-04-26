#!python
# coding: UTF-8
"""
author: kier
"""

import tensorstack as ts
import numpy


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


supported_map = {
    Name.Layer.nchw2nhwc: map_nchw2nchw_to_nchw
}

unsupported_set = {
    Name.Layer.nhwc2nchw,
    ts.Node.Parameter
}


def try_to_nchw(x):
    # type: (ts.Node) -> Union[ts.Node, None]
    op = x.op
    if op in supported_map:
        return supported_map[op](x)
    elif op in unsupported_set:
        return None
    else:
        raise NotImplementedError("{} was not marked in supported or unsupported".format(op))


def zipnode(x):
    # type: (ts.Node) -> ts.Node
    try_nchw_inputs = []
    for input in x.inputs:
        assert isinstance(input, ts.Node)
        if input.op == Name.Layer.nhwc2nchw:
            tmp = try_to_nchw(input.inputs[0])
            if tmp is not None:
                input = tmp
        try_nchw_inputs.append(input)
    zipped_inputs = []
    for input in try_nchw_inputs:
        zipped_inputs.append(zipnode(input))
    ts.Node.Link(x, zipped_inputs)

    if x.op == Name.Layer.nhwc2nchw:
        x.op = ts.zoo.Name.Layer.transpose
        x.params[ts.zoo.Name.permute] = numpy.asarray([0, 3, 1, 2], dtype=numpy.int32)
    elif x.op == Name.Layer.nchw2nhwc:
        x.op = ts.zoo.Name.Layer.transpose
        x.params[ts.zoo.Name.permute] = numpy.asarray([0, 2, 3, 1], dtype=numpy.int32)

    return x


def plot_graph(node):
    if not isinstance(node, (tuple, list)):
        node = [node,]
    for x in node:
        assert isinstance(x, ts.Node)
        plot_graph(x.inputs)
    for x in node:
        assert isinstance(x, ts.Node)
        print("{}: {} -> {}".format(x.op, [i.name for i in x.inputs], x.name))


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

    print()

    plot_graph(output)


