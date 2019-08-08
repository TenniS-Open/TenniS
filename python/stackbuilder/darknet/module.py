#!python
# coding: UTF-8
"""
author: kier
"""


from .darknet import Network
import tensorstack as ts
from .enum import *

from .layers.forward import *

import sys


map_layer_inference = {
    CONVOLUTIONAL: forward_convolutional_layer,
    MAXPOOL: forward_maxpool_layer,
    YOLO: forward_yolo_layer,
    ROUTE: forward_route_layer,
    UPSAMPLE: forward_upsample_layer,
    SHORTCUT: forward_shortcut_layer,
}


def darknet2module(net, yolo=True, out=None):
    # type: (Network, bool, Union[list[int], None]) -> ts.Module
    """

    :param net:
    :param yolo:
    :param out:
    :return:
    """

    '''
    pre-processor: bgr2rgb, to_float, scale(1/255.0), letterbox(net.w, net.h, 0.5), to_chw
    '''

    assert out is None or isinstance(out, (tuple, list))

    if yolo:
        sys.stderr.write("Module output all yolo layer\n")
        out = []

    if out is None:
        sys.stderr.write("Module output last non-cost layer\n")
        out = [-1]

    out = [net.n + x if x < 0 else x for x in out]

    if yolo:
        for i in range(net.n):
            if net.layers[i].type == YOLO:
                out.append(i)

    if len(out) == 0:
        raise Exception("Can not find any yolo layer, or given out index")

    input = ts.menu.param(name="_input", shape=(1, net.c, net.h, net.w))
    net.input = input

    for i in range(net.n):
        net.index = i
        l = net.layers[i]

        # set l.output to ts.Node
        type = l.type
        if l.type not in map_layer_inference:
            raise NotImplementedError("Not support layer: {}".format(l.type_string))

        output = map_layer_inference[type](l, net)
        if output is not None:
            l.output = output
        assert l.has("output") and l.output is not None

        net.input = l.output

    output_nodes = [net.layers[i].output for i in out]

    module = ts.Module()
    if yolo:
        yolo_nodes = output_nodes
        yolo_poster = ts.frontend.darknet.yolo_poster(name="yolo", x=input, inputs=yolo_nodes, thresh=0.5, nms=0.45)
        output_nodes = [yolo_poster]

    module.load(output_nodes)

    return module
