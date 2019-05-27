#!python
# coding: UTF-8
"""
author: kier
"""

from .parser import parse_network_cfg
from .loader import load_weights
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

    print net

    return None