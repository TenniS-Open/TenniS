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
        y = (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))

        bubble_inputs = [x, ]
        bubble_outputs = [y, ]

        ts_input = ts.menu.param(name="")
        ts_node = ts.zoo.tanh(name="tanh", x=ts_input)

        yield ts_node, bubble_inputs, bubble_outputs


def list_raw_norm_image_case():
    """
    yield image, epsilon
    :return:
    """
    import cv2
    image = cv2.imread("face.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yield image, 1e-5


def list_norm_image_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    for image, epsilon in list_raw_norm_image_case():
        import cv2
        x = numpy.asarray(image, dtype=numpy.float32)
        mean, std_dev = cv2.meanStdDev(x)
        new_m = mean[0][0]
        new_sd = std_dev[0][0]
        y = (x - new_m) / (new_sd + epsilon)

        print("mean={}, std_dev={}, epsilon={}".format(new_m, new_sd, epsilon))

        x = numpy.expand_dims(x, 0)
        y = numpy.expand_dims(y, 0)

        print("x.shape={}".format(x.shape))
        print("y.shape={}".format(y.shape))

        bubble_inputs = [x, ]
        bubble_outputs = [y, ]

        ts_input = ts.menu.param(name="")
        ts_node = ts.menu.op(name="norm_image", op_name="norm_image", inputs=[ts_input, ])
        ts_node.set("epsilon", epsilon, numpy.float32)

        yield ts_node, bubble_inputs, bubble_outputs


def list_raw_reduce_case():
    """
    yield data, dim, keep_dim, dtype
    :return:
    """
    numpy.random.seed(4482)
    data = numpy.random.rand(3, 4, 5, 6)
    data = data * 200 - 100
    yield data, 0, True, numpy.float32
    yield data, 1, True, numpy.float32
    yield data, 2, True, numpy.float32
    yield data, 3, True, numpy.float32
    yield data, -2, True, numpy.float32
    yield data, 0, False, numpy.float32
    yield data, 1, False, numpy.float32
    yield data, 2, False, numpy.float32
    yield data, 3, False, numpy.float32
    yield data, -2, False, numpy.float32


def list_reduce_sum_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    for data, dim, keep_dim, dtype in list_raw_reduce_case():
        x = numpy.asarray(data, dtype=dtype)
        y = numpy.sum(x, axis=dim, keepdims=keep_dim)

        bubble_inputs = [x, ]
        bubble_outputs = [y, ]

        ts_input = ts.menu.param(name="")
        ts_node = ts.zoo.reduce_sum(name="reduce_sum", x=ts_input, reduce_dims=dim, keep_dims=keep_dim)

        yield ts_node, bubble_inputs, bubble_outputs


def list_reduce_mean_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    for data, dim, keep_dim, dtype in list_raw_reduce_case():
        x = numpy.asarray(data, dtype=dtype)
        y = numpy.mean(x, axis=dim, keepdims=keep_dim)

        bubble_inputs = [x, ]
        bubble_outputs = [y, ]

        ts_input = ts.menu.param(name="")
        ts_node = ts.zoo.reduce_mean(name="reduce_mean", x=ts_input, reduce_dims=dim, keep_dims=keep_dim)

        yield ts_node, bubble_inputs, bubble_outputs


def list_raw_sqrt_case():
    """
    yield data, dtype
    :return:
    """
    numpy.random.seed(4482)
    data = numpy.random.rand(4, 3)
    data = data * 200 - 100
    yield data, numpy.float32
    yield data, numpy.float64


def list_sqrt_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    for data, dtype in list_raw_sqrt_case():
        x = numpy.asarray(data, dtype=dtype)
        y = numpy.sqrt(x).astype(dtype=dtype)

        bubble_inputs = [x, ]
        bubble_outputs = [y, ]

        ts_input = ts.menu.param(name="")
        ts_node = ts.zoo.sqrt(name="sqrt", x=ts_input)

        yield ts_node, bubble_inputs, bubble_outputs


def list_raw_tile_case():
    """
    yield x, repeats, dtype
    :return:
    """
    numpy.random.seed(4482)
    data = numpy.random.rand(2, 3)
    data = data * 200 - 100
    yield data, (2, 1), numpy.float32
    yield data, (3, 2), numpy.float32
    yield data, (2), numpy.float32
    yield data, (2, 1, 3), numpy.float32
    yield data, (0, 1), numpy.float32


def list_tile_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    for data, repeats, dtype in list_raw_tile_case():
        x = numpy.asarray(data, dtype=dtype)
        y = numpy.tile(x, repeats)

        print x.shape
        print repeats
        print y.shape

        bubble_inputs = [x, ]
        bubble_outputs = [y, ]

        ts_input = ts.menu.param(name="")
        ts_node = ts.zoo.tile(name="tile", repeats=repeats, x=ts_input)

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
    generate_func("norm_image", list_norm_image_case)
    generate_func("reduce_sum", list_reduce_sum_case)
    generate_func("reduce_mean", list_reduce_mean_case)
    generate_func("sqrt", list_sqrt_case)
    generate_func("tile", list_tile_case)