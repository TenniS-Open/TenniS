#!/usr/bin/env python

"""
Author: Kier
"""

from .. import Node
from .. import zoo
from .. import menu
from .. import device

import numpy


class Name(object):
    class Layer(object):
        shape_index_patch = "shape_index_patch"

    origin_patch = "origin_patch"
    origin = "origin"


def shape_index_patch(name, feat, pos, origin_patch, origin):
    """
    :param name: node name
    :param feat: feat node(blob)
    :param pos: pos node(blob)
    :param origin_patch: tuple(h, w)
    :param origin: tuple(h, w)
    :return:
    """
    assert isinstance(feat, Node)
    assert isinstance(pos, Node)

    # param
    origin_patch = zoo.to_const(origin_patch, "origin_patch")
    origin = zoo.to_const(origin, "origin")

    # operator
    node = menu.op(name=name, op_name=Name.Layer.shape_index_patch, inputs=[feat, pos])
    node.set(Name.origin_patch, origin_patch, numpy.int32)
    node.set(Name.origin, origin, numpy.int32)

    return node
