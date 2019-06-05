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


def make_upsample_layer(batch, h, w, c, stride):
    # type(int, int, int, int, int) -> Layer
    l = Layer()
    l.type = UPSAMPLE
    l.batch = batch
    l.w = w
    l.h = h
    l.c = c
    l.out_w = w*stride
    l.out_h = h*stride
    l.out_c = c
    l.reverse = 0
    if stride < 0:
        stride = -stride
        l.reverse = 1
        l.out_w = w/stride
        l.out_h = h/stride

    l.stride = stride
    l.outputs = l.out_w*l.out_h*l.out_c
    l.inputs = l.w*l.h*l.c
    l.delta =  None # calloc(l.outputs*batch, sizeof(float))
    l.output = None # calloc(l.outputs*batch, sizeof(float))

    # l.forward = forward_upsample_layer;
    # l.backward = backward_upsample_layer;

    if l.reverse:
        fprintf(sys.stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",
                stride, w, h, c, l.out_w, l.out_h, l.out_c)
    else:
        fprintf(sys.stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",
                stride, w, h, c, l.out_w, l.out_h, l.out_c)

    return l


def parse_upsample(options, params):
    stride = option_find_int(options, "stride", 2)
    l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride)
    l.scale = option_find_float_quiet(options, "scale", 1)
    return l
