#!/usr/bin/env python

"""
Author: Kier
"""

from .. import Node
from .. import zoo
from .. import menu
from .. import device

import numpy


def proposal(output_names,
             inputs,
             strides,
             ratios,
             scales,
             pre_nms_top_n=6000,
             post_nms_top_n=300,
             nms_thresh=0.7,
             min_size=16,
             min_level=2,
             max_level=5,
             canonical_scale=224,
             canonical_level=4):
    """
    :param output_names: (sequence of string) name of output
    :param inputs: (sequence of Tensor)  The inputs.
    :param strides: (sequence of int)  The strides of anchors.
    :param ratios: (sequence of float)  The ratios of anchors.
    :param scales: (sequence of float)  The scales of anchors.
    :param pre_nms_top_n: (int, optional, default=6000)  The number of anchors before nms.
    :param post_nms_top_n: (int, optional, default=300)  The number of anchors after nms.
    :param nms_thresh: (float, optional, default=0.7)  The threshold of nms.
    :param min_size: (int, optional, default=16)  The min size of anchors.
    :param min_level: (int, optional, default=2)  Finest level of the FPN pyramid.
    :param max_level: (int, optional, default=5)  Coarsest level of the FPN pyramid.
    :param canonical_scale: (int, optional, default=224)  The baseline scale of mapping policy.
    :param canonical_level: (int, optional, default=4)  Heuristic level of the canonical scale.
    :return: The proposals
    """

    for input in inputs:
        assert isinstance(input, Node)

    node = menu.op(name=output_names[0] + "_proposal", op_name="proposal", inputs=inputs)
    node.set("strides", strides, dtype=numpy.int32)
    node.set("ratios", ratios, dtype=numpy.float32)
    node.set("scales", scales, dtype=numpy.float32)

    node.set("pre_nms_top_n", pre_nms_top_n, dtype=numpy.int32)
    node.set("post_nms_top_n", post_nms_top_n, dtype=numpy.int32)
    node.set("nms_thresh", nms_thresh, dtype=numpy.float32)
    node.set("min_size", min_size, dtype=numpy.int32)
    node.set("min_level", min_level, dtype=numpy.int32)
    node.set("max_level", max_level, dtype=numpy.int32)
    node.set("canonical_scale", canonical_scale, dtype=numpy.int32)
    node.set("canonical_level", canonical_level, dtype=numpy.int32)

    return [menu.field(name=output_names[i], input=node, offset=i) for i in range(len(output_names))]


def roi_align(output_names,
              inputs,
              pool_h=0,
              pool_w=0,
              spatial_scale=1.0,
              sampling_ratio=2):
    """
    :param output_names: (sequence of string) name of output
    :param inputs: (sequence of Tensor) – The inputs, represent the Feature and RoIs respectively.
    :param pool_h: (int, optional, default=0) – The height of pooled tensor.
    :param pool_w: (int, optional, default=0) – The width of pooled tensor.
    :param spatial_scale: (float, optional, default=1.0) – The inverse of total down-sampling multiples on input tensor.
    :param sampling_ratio: (int, optional, default=2) – The number of sampling grids for each RoI bin.
    :return:
    """

    for input in inputs:
        assert isinstance(input, Node)

    node = menu.op(name=output_names[0] + "_roi_align", op_name="roi_align", inputs=inputs)
    node.set("pool_h", pool_h, dtype=numpy.int32)
    node.set("pool_w", pool_w, dtype=numpy.int32)

    node.set("spatial_scale", spatial_scale, dtype=numpy.float32)
    node.set("sampling_ratio", sampling_ratio, dtype=numpy.int32)

    return [menu.field(name=output_names[i], input=node, offset=i) for i in range(len(output_names))]