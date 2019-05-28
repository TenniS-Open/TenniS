#!python
# coding: UTF-8
"""
author: kier
"""

from .parser import parse_network_cfg
from .loader import load_weights
from .module import darknet2module
from . import darknet

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

    return module