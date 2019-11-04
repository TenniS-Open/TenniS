#!/usr/bin/env python

"""
:author Kier
"""

from typing import Union, List, Dict

import tensorstack as ts

if __name__ == "__main__":
    a = ts.menu.param("a", [3], ts.FLOAT32)
    b = ts.menu.param("b", [3], ts.FLOAT32)
    c = ts.menu.op("c", "add", [a, b])
    d = ts.menu.op("d", "add", [a, c])

    print(d)
    print(ts.inferer.infer(d))
    print(d)