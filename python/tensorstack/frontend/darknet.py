#!/usr/bin/env python

"""
Author: Kier
"""

from .. import Node
from .. import zoo
from .. import menu
from .. import device

import numpy


def yolo(name, x):
    node = menu.op(name=name, op_name="yolo", inputs=[x])

    return node