#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.mxnet.converter import convert


def test():
    convert("mxnet/model-finetune_nosoftmax", 1100000,
            {
                "data": [1, 3, 248, 248],
            },
            "mxnet.tsm")


if __name__ == '__main__':
    test()
