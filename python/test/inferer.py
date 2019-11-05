#!/usr/bin/env python

"""
:author Kier
"""

from typing import Union, List, Dict

import tensorstack as ts

if __name__ == "__main__":
    model = "/home/kier/git/TensorStack/python/test/caffe.tsm"

    with open(model, "rb") as f:
        module = ts.module.Module.Load(f)

    outputs = module.outputs

    for i in module.inputs:
        if not ts.inferer.has_infered(i):
            if not i.has(ts.Node.RetentionParam.dtype):
                i.dtype = ts.FLOAT32
            if not i.has(ts.Node.RetentionParam.shape):
                i.shape = [-1, 3, 256, 256]

    print(outputs[0])
    print(ts.inferer.infer(outputs[0]))
    print(outputs[0])