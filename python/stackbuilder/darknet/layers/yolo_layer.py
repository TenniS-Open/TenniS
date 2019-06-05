#!python
# coding: UTF-8
"""
author: kier
"""

from ..enum import *
from ..config import *
from ..darknet import Layer
from ..darknet import calloc

import sys
import math


def make_yolo_layer(batch, h, w, n, total, mask, classes):
    # type(int, int, int, int, int, list[int], int) -> Layer
    l = Layer()
    l.type = YOLO

    l.n = n
    l.total = total
    l.batch = batch
    l.h = h
    l.w = w
    l.c = n*(classes + 4 + 1)
    l.out_w = l.w
    l.out_h = l.h
    l.out_c = l.c
    l.classes = classes
    l.cost = calloc(1, float)
    l.biases = calloc(total * 2, float)
    if mask is not None and len(mask) > 0:
        l.mask = mask
    else:
        l.mask = range(n)

    l.bias_updates = calloc(n*2, float)
    l.outputs = h*w*n*(classes + 4 + 1)
    l.inputs = l.outputs
    l.truths = 90*(4 + 1)
    l.delta = calloc(batch*l.outputs, float)
    l.output = calloc(batch*l.outputs, float)
    l.biases = [0.5] * (total * 2)

    # l.forward = forward_yolo_layer;
    # l.backward = backward_yolo_layer;

    sys.stderr.write("yolo\n")

    return l


def read_map(filename):
    # type: (filename) -> list
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [s.strip() for s in lines]
    return [int(s) for s in lines]


def parse_yolo(options, params):
    # type: (Session, SizeParams) -> Layer
    classes = option_find_int(options, "classes", 20)
    total = option_find_int(options, "num", 1)
    num = total

    mask = option_find_int_list(options, "mask", [])
    num = len(mask)
    l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes)
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max",90)
    l.jitter = option_find_float(options, "jitter", .2)

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5)
    l.truth_thresh = option_find_float(options, "truth_thresh", 1)
    l.random = option_find_int_quiet(options, "random", 0)

    map_file = option_find_str(options, "map", "")
    if map_file is not None and len(map_file) > 0:
        l.map = read_map(map_file)
    else:
        l.map = None

    a = option_find_float_list(options, "anchors",  [])
    if a is not None and len(a) > 0:
        l.biases = a

    return l