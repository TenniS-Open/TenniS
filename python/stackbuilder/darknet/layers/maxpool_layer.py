#!python
# coding: UTF-8
"""
author: kier
"""

from ..enum import *
from ..config import *
from ..darknet import Layer

import sys
import math


def make_maxpool_layer(batch, h, w, c, size, stride, padding):
    l = Layer()
    l.type = MAXPOOL

    l.batch = batch
    l.h = h
    l.w = w
    l.c = c
    l.pad = padding
    l.out_w = (w + padding - size)//stride + 1
    l.out_h = (h + padding - size)//stride + 1
    l.out_c = c
    l.outputs = l.out_h * l.out_w * l.out_c
    l.inputs = h*w*c
    l.size = size
    l.stride = stride
    # output_size = l.out_h * l.out_w * l.out_c * batch
    l.indexes = None # calloc(output_size, sizeof(int));
    l.output =  None # calloc(output_size, sizeof(float));
    l.delta =   None # calloc(output_size, sizeof(float));
    # l.forward = forward_maxpool_layer;
    # l.backward = backward_maxpool_layer;

    sys.stderr.write("max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n" %
                     (size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c))

    return l


def parse_max_pool(options, params):
    # type: (Session, SizeParams) -> Layer
    stride = option_find_int(options, "stride",1)
    size = option_find_int(options, "size",stride)
    padding = option_find_int_quiet(options, "padding", size-1)

    h = params.h
    w = params.w
    c = params.c
    batch=params.batch
    if h == 0 or w == 0 or c == 0:
        raise NotImplementedError("Layer before convolutional layer must output image.")

    layer = make_maxpool_layer(batch,h,w,c,size,stride,padding)
    return layer