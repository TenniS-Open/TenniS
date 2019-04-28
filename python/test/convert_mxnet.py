#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.mxnet.converter import convert
import tensorstack as ts


def test():
    convert("mxnet/resnet50/model-finetune_nosoftmax", 1100000,
            "mxnet-resnet50.tsm",
            input_nodes={
                "data": ts.menu.param("data", [1, 3, 248, 248]),
            },
            output_node_names=['fc1_act', ]
            )


if __name__ == '__main__':
    test()
