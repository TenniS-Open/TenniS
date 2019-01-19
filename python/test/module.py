#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from tensorstack import Module
from tensorstack import bubble
from tensorstack import tensor
from tensorstack import device


def test():
    a = bubble.param("a")
    b = bubble.param("b")
    data = bubble.data("data", tensor.from_any(3, dtype=float), device=device.CPU)

    c = bubble.op("c", "sum", [a, b, data])

    module = Module()

    module.load(c)

    with open("test.module.txt", "wb") as fo:
        Module.Save(stream=fo, module=module)

    with open("test.module.txt", "rb") as fi:
        module = Module.Load(stream=fi)

    with open("test.module.txt", "wb") as fo:
        Module.Save(stream=fo, module=module)

    with open("test.module.txt", "rb") as fi:
        module = Module.Load(stream=fi)

    with open("test.module.txt", "wb") as fo:
        Module.Save(stream=fo, module=module)


if __name__ == '__main__':
    test()
