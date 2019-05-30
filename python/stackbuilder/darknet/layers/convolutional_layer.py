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


def convolutional_out_height(l):
    # type: (Layer) -> int
    return (l.h + 2*l.pad - l.size) / l.stride + 1


def convolutional_out_width(l):
    # type: (Layer) -> int
    return (l.w + 2*l.pad - l.size) / l.stride + 1


def make_convolutional_layer(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize,
                             binary, xnor, adam):
    l = Layer()
    l.type = CONVOLUTIONAL

    l.groups = groups
    l.h = h
    l.w = w
    l.c = c
    l.n = n
    l.binary = binary
    l.xnor = xnor
    l.batch = batch
    l.stride = stride
    l.size = size
    l.pad = padding
    l.batch_normalize = batch_normalize

    l.weights = None        # calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = None # calloc(c/groups*n*size*size, sizeof(float));

    l.biases = None         # calloc(n, sizeof(float));
    l.bias_updates = None   # calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size
    l.nbiases = n

    # float scale = 1./sqrt(size*size*c);
    scale = math.sqrt(2./(size*size*c/l.groups))
    # printf("convscale %f\n", scale);
    # scale = .02;
    # for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    # for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    out_w = convolutional_out_width(l)
    out_h = convolutional_out_height(l)
    l.out_h = out_h
    l.out_w = out_w
    l.out_c = n
    l.outputs = l.out_h * l.out_w * l.out_c
    l.inputs = l.w * l.h * l.c

    if binary != 0:
        raise NotImplementedError("Not supporting convert with binary=1")
    if xnor != 0:
        raise NotImplementedError("Not supporting convert with xnor=1")
    if batch_normalize != 0:
        l.scales = None
        l.scale_updates = None
        l.mean = None
        l.variance = None
        l.mean_delta = None
        l.variance_delta = None
        l.rolling_mean = None
        l.rolling_variance = None
        l.x = None
        l.x_norm = None

    """
    No memry alloc code
    """

    l.output = None
    l.delta = None
    l.mean = None
    l.variance = None
    l.mean_delta = None
    l.variance_delta = None
    l.rolling_mean = None
    l.rolling_variance = None
    l.x = None
    l.x_norm = None

    if adam != 0:
        l.m = None
        l.v = None
        l.bias_m = None
        l.scale_m = None
        l.bias_v = None
        l.scale_v = None

    l.workspace_size = 0
    l.activation = activation

    sys.stderr.write("conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n" %
                     (n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c,
                      (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.))

    return l


def parse_convolutional(options, params):
    # type: (Session, SizeParams) -> Layer
    n = option_find_int(options, "filters",1)
    size = option_find_int(options, "size",1)
    stride = option_find_int(options, "stride",1)
    pad = option_find_int_quiet(options, "pad",0)
    padding = option_find_int_quiet(options, "padding",0)
    groups = option_find_int_quiet(options, "groups", 1)
    if pad != 0:
        padding = size/2

    activation_s = option_find_str(options, "activation", "logistic")
    activation = get_activation(activation_s)

    h = params.h
    w = params.w
    c = params.c
    batch=params.batch
    if h == 0 or w == 0 or c == 0:
        raise NotImplementedError("Layer before convolutional layer must output image.")
    batch_normalize = option_find_int_quiet(options, "batch_normalize", 0)
    binary = option_find_int_quiet(options, "binary", 0)
    xnor = option_find_int_quiet(options, "xnor", 0)

    layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize,
                                     binary, xnor, params.net.adam)

    layer.flipped = option_find_int_quiet(options, "flipped", 0)
    layer.dot = option_find_float_quiet(options, "dot", 0)

    return layer
