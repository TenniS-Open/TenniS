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
from ..darknet import fprintf

import sys
import math


def make_shortcut_layer(batch, index, w, h, c, w2, h2, c2):
    # type(int, int, int, int, int, int, int, int) -> Layer
    fprintf(sys.stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c)

    l = Layer()
    l.type = SHORTCUT
    l.batch = batch
    l.w = w2
    l.h = h2
    l.c = c2
    l.out_w = w
    l.out_h = h
    l.out_c = c
    l.outputs = w*h*c
    l.inputs = l.outputs

    l.index = index

    l.delta = None  # calloc(l.outputs*batch, sizeof(float));
    l.output = None # calloc(l.outputs*batch, sizeof(float));;

    # l.forward = forward_shortcut_layer;
    # l.backward = backward_shortcut_layer;

    return l


def parse_shortcut(options, params):
    # type: (Session, SizeParams) -> Layer
    index = option_find_int(options, "from", -1)
    if index < 0:
        index = params.index + index

    net = params.net

    batch = params.batch
    layer_f = net.layers[index]

    layer_s = make_shortcut_layer(batch, index, params.w, params.h, params.c,
                                  layer_f.out_w, layer_f.out_h, layer_f.out_c)

    activation_s = option_find_str(options, "activation", "linear")
    activation = get_activation(activation_s)
    layer_s.activation = activation
    layer_s.alpha = option_find_float_quiet(options, "alpha", 1)
    layer_s.beta = option_find_float_quiet(options, "beta", 1)
    return layer_s
