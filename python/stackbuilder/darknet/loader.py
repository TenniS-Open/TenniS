#!python
# coding: UTF-8
"""
author: kier
"""


from .darknet import Network
from .darknet import Layer
from .darknet import fprintf
from sys import stderr

from .param import *
from .enum import *

import numpy


def load_float_tensor(fp, shape):
    count = shape
    if isinstance(shape, (tuple, list)):
        count = numpy.prod(shape)
    bytes = fp.read(count * 4)
    dtype_numpy = numpy.dtype(numpy.float32)
    dtype_numpy = dtype_numpy.newbyteorder('<')
    tensor = numpy.frombuffer(bytes, dtype=dtype_numpy)
    if isinstance(shape, (tuple, list)):
        tensor = numpy.resize(tensor, new_shape=shape)
    return tensor


def transpose_matrix(a, rows, cols):
    # type: (numpy.ndarray, int, int) ->numpy.ndarray
    a = numpy.reshape(a, newshape=(rows, cols))
    a = a.transpose()
    return a


def load_convolutional_weights(l, fp):
    # type: (Layer, file) -> None
    if l.binary:
        raise NotImplementedError("can not load weights: binary")
    if l.numload:
        l.n = l.numload
    num = l.c // l.groups * l.n * l.size * l.size
    l.biases = load_float_tensor(fp, l.n)
    if l.batch_normalize and not l.dontloadscales:
        l.scales = load_float_tensor(fp, l.n)
        l.rolling_mean = load_float_tensor(fp, l.n)
        l.rolling_variance = load_float_tensor(fp, l.n)

    l.weights = load_float_tensor(fp, num)
    if l.flipped:
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n)


def load_batchnorm_weights(l, fp):
    # type: (Layer, file) -> None
    l.scales = load_float_tensor(fp, l.c)
    l.rolling_mean = load_float_tensor(fp, l.c)
    l.rolling_variance = load_float_tensor(fp, l.c)


def load_connected_weights(l, fp, transpose):
    # type: (Layer, file, bool) -> None
    l.biases = load_float_tensor(fp, l.outputs)
    l.weights = load_float_tensor(fp, l.outputs*l.inputs)
    if transpose:
        transpose_matrix(l.weights, l.inputs, l.outputs)

    if l.batch_normalize and not l.dontloadscales:
        l.scales = load_float_tensor(fp, l.outputs)
        l.rolling_mean = load_float_tensor(fp, l.outputs)
        l.rolling_variance = load_float_tensor(fp, l.outputs)


def load_weights_upto(net, filename, start, cutoff):
    # type: (Network, str, int, int) -> None

    fprintf(stderr, "Loading weights from %s...", filename)
    stderr.flush()

    with open(filename, "rb") as fp:
        major, minor, revision = read_param(fp, Int32, Int32, Int32)

        if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
            net.seen, = read_param(fp, Uint64)
        else:
            net.seen, = read_param(fp, Int32)

        transpose = major > 1000 or minor > 1000

        for i in range(start, min(net.n, cutoff)):
            l = net.layers[i]
            if l.dontload:
                continue

            if l.type == CONVOLUTIONAL or l.type == DECONVOLUTIONAL:
                load_convolutional_weights(l, fp)

            if l.type == CONNECTED:
                load_connected_weights(l, fp, transpose)

            if l.type == BATCHNORM:
                load_batchnorm_weights(l, fp)

            if l.type == CRNN:
                load_convolutional_weights(l.input_layer, fp)
                load_convolutional_weights(l.self_layer, fp)
                load_convolutional_weights(l.output_layer, fp)

            if l.type == RNN:
                load_connected_weights(l.input_layer, fp, transpose)
                load_connected_weights(l.self_layer, fp, transpose)
                load_connected_weights(l.output_layer, fp, transpose)

            if l.type == LSTM:
                load_connected_weights(l.wi, fp, transpose)
                load_connected_weights(l.wf, fp, transpose)
                load_connected_weights(l.wo, fp, transpose)
                load_connected_weights(l.wg, fp, transpose)
                load_connected_weights(l.ui, fp, transpose)
                load_connected_weights(l.uf, fp, transpose)
                load_connected_weights(l.uo, fp, transpose)
                load_connected_weights(l.ug, fp, transpose)

            if l.type == GRU:
                load_connected_weights(l.wz, fp, transpose)
                load_connected_weights(l.wr, fp, transpose)
                load_connected_weights(l.wh, fp, transpose)
                load_connected_weights(l.uz, fp, transpose)
                load_connected_weights(l.ur, fp, transpose)
                load_connected_weights(l.uh, fp, transpose)

            if l.type == LOCAL:
                locations = l.out_w*l.out_h
                size = l.size*l.size*l.c*l.n*locations
                l.biases, = read_param(fp, [Float] * l.outputs)
                l.weights, = read_param(fp, [Float] * size)

    fprintf(stderr, "Done!\n")


def load_weights(net, filename):
    # type: (Network, str) -> None
    load_weights_upto(net, filename, 0, net.n)