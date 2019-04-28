#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

# from stackbuilder.torch.converter import convert
from stackbuilder.torch.converter import convert_by_onnx


def test():
    convert_by_onnx("cpu_checkpoint.pkl", "test.onnx.jit.tsm", [(1, 3, 224, 224)])


if __name__ == '__main__':
    test()
