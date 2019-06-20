"""
@author: Kier
"""


import tensorstack as ts
import numpy
import os

import math


def write_case(path, bubble, inputs, outputs):
    # type: (str, ts.Node, List[numpy.ndarray], List[numpy.ndarray]) -> None
    print("Writing: {}".format(path))
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(os.path.join(path, "0.{}.txt".format(bubble.op)), "w") as f:
        f.write("{}\n{}\n{}\n".format(len(bubble.params), len(inputs), len(outputs)))

    for param, value in bubble.params.iteritems():
        with open(os.path.join(path, "1.{}.t".format(param)), "wb") as f:
            ts.tensor.write_tensor(f, value)

    for i in range(len(inputs)):
        input = inputs[i]
        with open(os.path.join(path, "2.input_{}.t".format(i)), "wb") as f:
            ts.tensor.write_tensor(f, input)

    for i in range(len(outputs)):
        output = outputs[i]
        with open(os.path.join(path, "3.output_{}.t".format(i)), "wb") as f:
            ts.tensor.write_tensor(f, output)


def list_raw_abs_case():
    """
    yield data, dtype
    :return:
    """
    numpy.random.seed(4482)
    data = numpy.random.rand(4, 3)
    data = data * 200 - 100
    yield data, numpy.float32
    yield data, numpy.float64
    yield data, numpy.int32


def list_abs_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    for data, dtype in list_raw_abs_case():
        x = numpy.asarray(data, dtype=dtype)
        y = numpy.abs(x)

        bubble_inputs = [x, ]
        bubble_outputs = [y, ]

        ts_input = ts.menu.param(name="")
        ts_node = ts.zoo.abs(name="abs", x=ts_input)

        yield ts_node, bubble_inputs, bubble_outputs


def list_raw_tanh_case():
    """
    yield data, dtype
    :return:
    """
    numpy.random.seed(4482)
    data = numpy.random.rand(32, 64)
    data = data * 20 - 10
    yield data, numpy.float32
    data = numpy.random.rand(64, 32)
    data = data * 20 - 10
    yield data, numpy.float32


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def list_tanh_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    for data, dtype in list_raw_tanh_case():
        x = numpy.asarray(data, dtype=dtype)
        y = numpy.exp(x) - numpy.exp(-x) / (numpy.exp(x) + numpy.exp(-x))

        bubble_inputs = [x, ]
        bubble_outputs = [y, ]

        ts_input = ts.menu.param(name="")
        ts_node = ts.zoo.tanh(name="tanh", x=ts_input)

        yield ts_node, bubble_inputs, bubble_outputs


def generate_func(path, func):
    """
    generate test case
    :param path: path to write case
    :param func: func return iter
    :return:
    """
    iter = 1
    for bubble, inputs, outputs in func():
        case_path = os.path.join(path, str(iter))

        write_case(case_path, bubble, inputs, outputs)

        iter += 1


if __name__ == '__main__':
    generate_func("abs", list_abs_case)
    generate_func("tanh", list_tanh_case)