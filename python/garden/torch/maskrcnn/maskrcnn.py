#!/usr/bin/env python

"""
Author: Kier
"""

from tensorstack import Node
from tensorstack import zoo
from tensorstack import menu
from tensorstack import device
from tensorstack import tensor

import numpy
import math


class Name(object):
    class Layer(object):
        anchor_generator = "maskrcnn:anchor_generator"
        box_selector = "maskrcnn:box_selector"


def anchor_generator(name, images, features, strides, cell_anchors):
    """

    :param name:
    :param images: Packed[image, image's shape]
    :param features: Packed[feature_maps...]
    :param strides: List of int
    :param cell_anchors: list of Matrix
    :return: Repeating [ListAnchorNumber, [image_width, image_height], Anchor[-1, 4], ...]
    """
    assert isinstance(images, Node)
    assert isinstance(features, Node)

    cell_anchors = [tensor.from_any(base_anchors, numpy.float32) for base_anchors in cell_anchors]

    node = menu.op(name=name, op_name=Name.Layer.anchor_generator, inputs=[images, features])
    node.set("strides", strides, numpy.int32)
    node.set("cell_anchors", tensor.PackedTensor(cell_anchors), numpy.float32)

    return node


def box_selector(name, anchors, objectness, rpn_box_regression,
                 pre_nms_top_n,
                 min_size,
                 nms_thresh,
                 post_nms_top_n,
                 fpn_post_nms_top_n,
                 weights=None,
                 bbox_xform_clip=math.log(1000. / 16)):
    """

    :param name:
    :param anchors: Repeating [ListAnchorNumber, [image_width, image_height], Anchor[-1, 4], ...]
    :param objectness: PackedTensor for each scale objectness
    :param rpn_box_regression: PackedTensor for each scale
    :return: Repeating [[image_width, image_height], Boxes[-1, 4], Scores]
    """
    if weights is None:
        weights = (1.0, 1.0, 1.0, 1.0)

    node = menu.op(name=name, op_name=Name.Layer.box_selector, inputs=[anchors, objectness, rpn_box_regression])
    node.set("pre_nms_top_n", pre_nms_top_n, numpy.int32)

    node.set("weights", weights, numpy.float32)
    node.set("bbox_xform_clip", bbox_xform_clip, numpy.float32)

    node.set("min_size", min_size, numpy.int32)
    node.set("nms_thresh", nms_thresh, numpy.float32)
    node.set("post_nms_top_n", post_nms_top_n, numpy.int32)

    node.set("fpn_post_nms_top_n", fpn_post_nms_top_n, numpy.int32)

    return node
