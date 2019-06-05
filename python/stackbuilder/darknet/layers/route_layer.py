#!python
# coding: UTF-8
"""
author: kier
"""

from ..enum import *
from ..config import *
from ..darknet import Layer
from ..darknet import calloc
from ..darknet import SizeParams

import sys
import math


def make_route_layer(batch, n, input_layers, input_sizes):
    # type(int, int, list[int], list[int]) -> Layer
    sys.stderr.write("route ")

    l = Layer()
    l.type = ROUTE
    l.batch = batch
    l.n = n
    l.input_layers = input_layers
    l.input_sizes = input_sizes
    outputs = 0
    for i in range(n):
        sys.stderr.write(" %d" % (input_layers[i]))
        outputs += input_sizes[i]

    sys.stderr.write("\n")

    l.outputs = outputs
    l.inputs = outputs
    l.delta =  None # calloc(outputs*batch, sizeof(float));
    l.output = None # calloc(outputs*batch, sizeof(float));;

    # l.forward = forward_route_layer;
    # l.backward = backward_route_layer;

    return l


def parse_route(options, params):
    # type: (Session, SizeParams) -> Layer
    l = option_find_int_list(options, "layers", [])
    if l is None or len(l) == 0:
        raise Exception("Route Layer must specify input layers")
    n = len(l)
    layers = l
    layers = [params.index + index if index < 0 else index for index in layers]
    sizes = [params.net.layers[index].outputs for index in layers]

    batch = params.batch

    layer = make_route_layer(batch, n, layers, sizes)
    net = params.net

    first = params.net.layers[layers[0]]
    layer.out_w = first.out_w
    layer.out_h = first.out_h
    layer.out_c = first.out_c
    for i in range(1, n):
        index = layers[i]
        iter = net.layers[index]
        if iter.out_w == first.out_w and iter.out_h == first.out_h:
            layer.out_c += iter.out_c
        else:
            layer.out_h = layer.out_w = layer.out_c = 0

    return layer
