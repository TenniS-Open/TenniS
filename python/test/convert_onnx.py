#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.onnx.converter import convert


def test():
    convert("torch_model.onnx", "test.tsm")


if __name__ == '__main__':
    test()
