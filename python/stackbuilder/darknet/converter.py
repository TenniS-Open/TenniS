#!python
# coding: UTF-8
"""
author: kier
"""

from .parser import parse_network_cfg
from .loader import load_weights
from .module import darknet2module
from . import darknet
import tensorstack as ts

import sys


def convert(cfg_file, weights_file, output_file):
    """
    Darknet model converter
    :param cfg_file: cfg file
    :param weights_file: weights file
    :param output_file: output tsm file
    :return:
    """

    net = parse_network_cfg(cfg_file)
    load_weights(net, weights_file)

    darknet.set_batch_network(net, 1)

    module = darknet2module(net)

    with open(output_file, "wb") as fo:
        ts.Module.Save(stream=fo, module=module)

    print("============ Summary ============")
    print("Input cfg: {}".format(cfg_file))
    print("Input weihts: {}".format(weights_file))
    print("Output file: {}".format(output_file))
    index = 0
    print("Input node: ")
    for node in module.inputs:
        assert isinstance(node, ts.Node)
        print("{}: {}, shape={}".format(index, node.name, node.shape))
        index += 1
    index = 0
    print("Output node: ")
    for node in module.outputs:
        assert isinstance(node, ts.Node)
        print("{}: {}".format(index, node.name))
        index += 1

    return module