"""
@author: Kier
"""


import tensorstack as ts
import numpy
import os


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


def list_raw_case():
    """
    yield([data, begin, end, stride])
    :return:
    """
    data = [[[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]]]
    yield data, [0, 0, 0], [1, 1, 1], None
    yield data, [0, 0, 0], [-1, -1, -1], None
    yield data, [0, 0, 0], [-1, -1, -1], [2, 2, 2]
    yield data, [0, 0, 0], [3, 3, 3], [2, 2, 2]
    yield data, [0, 0], [3, 2], [2, 1]


def list_raw_pack_case():
    """
    yield(data, axis)
    :return:
    """
    data = [[1, 4], [2, 5], [3, 6]]
    yield data, 0
    yield data, 1
    yield data, -1
    yield data, -2


def list_pack_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    import tensorflow as tf

    with tf.Session() as sess:
        for data, axis in list_raw_pack_case():
            x = tf.stack(data, axis=axis)
            y = sess.run(x)

            bubble_inputs = [numpy.asarray(item, dtype=numpy.float32) for item in data]
            bubble_outputs = [numpy.asarray(y, dtype=numpy.float32), ]

            ts_inputs = [ts.menu.data(name="", value=item) for item in data]

            ts_node = ts.frontend.tf.stack("stack", tensors=ts_inputs, axis=axis)

            yield ts_node, bubble_inputs, bubble_outputs


def list_case():
    """
    yield bubble, inputs, outputs
    :return:
    """
    import tensorflow as tf

    with tf.Session() as sess:
        for data, begin, end, strides in list_raw_case():
            x = tf.strided_slice(data, begin=begin, end=end, strides=strides)
            y = sess.run(x)

            slice_data = ts.menu.data(name="data", value=data)
            slice = ts.frontend.tf.strided_slice(
                name="stride_slice", x=slice_data, begin=begin, end=end, stride=strides)

            data = numpy.asarray(data, dtype=numpy.float32)
            y = numpy.asarray(y, dtype=numpy.float32)

            yield slice, [data, ], [y, ]


def generate_strided_slice(path):
    """
    generate test case
    :param path: path to write case
    :return:
    """
    iter = 1
    for bubble, inputs, outputs in list_case():
        case_path = os.path.join(path, str(iter))

        write_case(case_path, bubble, inputs, outputs)

        iter += 1


def generate_stack(path):
    """
    generate test case
    :param path: path to write case
    :return:
    """
    iter = 1
    for bubble, inputs, outputs in list_pack_case():
        case_path = os.path.join(path, str(iter))

        write_case(case_path, bubble, inputs, outputs)

        iter += 1


def list_raw_l2_norm():
    """
    yield([data, dim, epsilon])
    :return:
    """
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    yield data, 1, 1e-10


def list_l2_norm():
    """
    yield bubble, inputs, outputs
    :return:
    """
    for data, dim, epsilon in list_raw_l2_norm():
        x = numpy.asarray(data, dtype=numpy.float32)
        N = x.shape[0]
        y = numpy.zeros_like(x)
        for i in range(N):
            y[i, :] = x[i, :] / numpy.sqrt(numpy.sum(x[i, :] ** 2) + epsilon)

        l2_norm_data = ts.menu.data(name="data", value=data)
        l2_norm = ts.zoo.l2_norm(
            name="l2_norm", x=l2_norm_data, dim=dim, epsilon=epsilon)

        data = numpy.asarray(data, dtype=numpy.float32)
        y = numpy.asarray(y, dtype=numpy.float32)

        yield l2_norm, [data, ], [y, ]


def generate_l2_norm(path):
    """
    generate test case
    :param path: path to write case
    :return:
    """
    iter = 1
    for bubble, inputs, outputs in list_l2_norm():
        case_path = os.path.join(path, str(iter))

        write_case(case_path, bubble, inputs, outputs)

        iter += 1


if __name__ == '__main__':
    generate_strided_slice("strided_slice")
    generate_stack("stack")
    generate_l2_norm("l2_norm")