#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.darknet.converter import convert
import tensorstack as ts


def test():
    convert("/Users/seetadev/Documents/SDK/CLion/darknet/bin/yolov3.cfg",
            "/Users/seetadev/Documents/SDK/CLion/darknet/bin/yolov3.weights",
            "yolov3.tsm")


if __name__ == '__main__':
    test()
