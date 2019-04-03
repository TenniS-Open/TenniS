#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.vvvv.converter import convert
from stackbuilder.vvvv import param
import tensorstack as ts


def test():
    with open("net.dat", "rb") as stream:
        header = param.read_param(stream, param.Int, param.Int, param.Int, param.Int, param.Float, param.Float, param.Float)
        print header
        input_channels = header[0]
        input_height = header[1]
        input_width = header[2]
        input = ts.menu.param("_input", shape=(1, input_channels, input_height, input_width))
        convert(stream, "test.vvvv.tsm", inputs=[input, ])


if __name__ == '__main__':
    test()
