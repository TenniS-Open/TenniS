#!python
# coding: UTF-8
"""
author: kier
"""

import mxnet as mx
import json

from . import parser

import tensorstack as ts


def convert(model_prefix, epoch, input_shape, output_file):
    """
    :param model_prefix: string of model prefix
    :param epoch: int
    :param input_shape: dict of string to shape, like {"data", [1, 3, 248, 248]}.
    :param output_file: path to output file
    :return:
    """
    symbol_json = '%s-symbol.json' % (model_prefix, )
    symbol = None
    with open(symbol_json) as f:
        symbol = json.load(f)

    _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)

    graph = parser.Graph(symbol, arg_params, aux_params)

    def convert_null(node, inputs):
        assert len(inputs) == 0
        name = node["name"]
        if name in input_shape:
            print("--# -=[ Placeholder: {}, {} ]=-".format(name, input_shape[name]))
            return ts.menu.param(name, input_shape[name])
        param = graph.param(name)
        if param is not None:
            param = param.asnumpy()
            print("--# -=[ Load data: {}, {} ]=-".format(name, param.shape))
            return ts.menu.data(name, param)
        raise Exception("Can not load param: {}".format(name))

    def convert_not_implemented(node, inputs):
        # type: (object, List[ts.Node]) -> ts.Node
        print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

        raise NotImplementedError(node["op"])

    # function format(node, inputs)
    converter_map = {
        "null": convert_null,
        # add layer converter here
        "BatchNorm": convert_not_implemented,
    }

    ts_nodes = [None] * len(graph.nodes)

    def convert_at(i):
        if ts_nodes[i] is not None:
            return
        node = graph.nodes[i]
        # name = node["name"]
        op = node["op"]
        if op not in converter_map:
            raise Exception("Not supported Layer: {}".format(op))
        converter = converter_map[op]
        inputs = [ input[0] for input in node["inputs"] ]
        for input in inputs:
            assert input != i
            # TODO: checking loop
            convert_at(input)
        inputs = [ ts_nodes[input] for input in inputs ]
        ts_nodes[i] = converter(node, inputs)

    for i in range(len(graph.nodes)):
        convert_at(i)

    outputs = [ts_nodes[head] for head in graph.heads]

    pass