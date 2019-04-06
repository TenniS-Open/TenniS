#!python
# coding: UTF-8
"""
author: kier
"""

import tensorstack as ts

import tensorflow as tf
import numpy


def convert(graph, inputs, outputs, output_file):
    if inputs is None:
        raise Exception("param #2 inputs must be set.")
    if outputs is None:
        raise Exception("param #3 outputs must be set.")
    if not isinstance(inputs, list) or not isinstance(inputs, tuple):
        inputs = (inputs, )
    if not isinstance(outputs, list) or not isinstance(outputs, tuple):
        outputs = (outputs, )

    # function format(node, inputs)
    map_converter = {
        # add layer converter here
        "Identity": convert_identity,
        "ConcatV2": convert_concat_v2,
        "StridedSlice": convert_not_implemented,
        "Reshape": convert_reshape,
        "Pack": convert_not_implemented,
        "Pad": convert_not_implemented,
        "GatherV2": convert_not_implemented,
        "Sub": convert_not_implemented,
        "Add": convert_not_implemented,
        "Mul": convert_not_implemented,
        "BiasAdd": convert_not_implemented,
        "Conv2D": convert_not_implemented,
        "Relu": convert_not_implemented,
        "AvgPool": convert_not_implemented,
        "RealDiv": convert_not_implemented,
        "Placeholder": convert_placeholder,
        "Const": convert_const,
    }

    map_tf_node_ts_node = {}

    output_ts_nodes = []

    def convert_node(tf_node):
        if tf_node in map_tf_node_ts_node:
            return map_tf_node_ts_node[tf_node]
        node_op = tf_node.op
        node_op_type = node_op.type
        if node_op_type not in map_converter:
            raise Exception("Not supported Layer: {}".format(node_op_type))
        converter = map_converter[node_op_type]

        input_ts_nodes = []
        for input in node_op.inputs:
            assert input != tf_node
            # TODO: checking loop
            input_ts_nodes.append(convert_node(input))

        map_tf_node_ts_node[tf_node] = converter(tf_node, input_ts_nodes)

    for output in outputs:
        output_ts_nodes.append(convert_node(output))

    print output_ts_nodes

    # checking inputs
    input_ts_nodes = []
    for input in inputs:
        if input not in map_tf_node_ts_node:
            raise Exception("Node {} not in graph".format(input.op.name))
        input_ts_nodes.append(map_tf_node_ts_node[input])

    print input_ts_nodes
    pass


def convert_identity(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(tf_node.op.type, tf_node.op.name))
    assert len(inputs) == 1
    return inputs[0]


def convert_concat_v2(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(tf_node.op.type, tf_node.op.name))
    assert len(inputs) > 1
    N = tf_node.op.get_attr('N')

    if N != len(inputs) - 1:
        raise NotImplementedError("Concat N={} with {} inputs".format(N, len(inputs) - 1))

    axis = tf_node.op.inputs[N].eval()
    axis = numpy.asarray(axis, dtype=numpy.int32)

    if axis < 0 or axis >= 4:
        raise NotImplementedError("Concat axis: {}".format(axis))

    return ts.zoo.concat(tf_node.op.name, inputs[:-1], dim=axis)


def convert_strided_slice(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(tf_node.op.type, tf_node.op.name))

    raise NotImplementedError(tf_node.op.type)


def convert_reshape(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(tf_node.op.type, tf_node.op.name))
    assert len(inputs) == 2

    shape = tf_node.op.inputs[1].eval()
    shape = numpy.asarray(shape, dtype=numpy.int32)

    return ts.zoo.reshape(tf_node.op.name, inputs[0], shape=shape)


def convert_not_implemented(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(tf_node.op.type, tf_node.op.name))

    raise NotImplementedError(tf_node.op.type)


def convert_placeholder(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(tf_node.op.type, tf_node.op.name))

    assert len(inputs) == 0

    return ts.menu.param(name=tf_node.op.name)


def convert_const(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(tf_node.op.type, tf_node.op.name))

    assert len(inputs) == 0

    return ts.menu.data(name=tf_node.op.name, value=tf_node.eval())
