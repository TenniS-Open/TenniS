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
    data = [[[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]]
    yield data, [0, 0, 0], [1, 1, 1], None
    yield data, [0, 0, 0], [-1, -1, -1], None
    yield data, [0, 0, 0], [-1, -1, -1], [2, 2, 2]
    yield data, [0, 0, 0], [3, 3, 3], [2, 2, 2]
    yield data, [0, 0], [3, 2], [2, 1]


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

            yield slice, [data, ], [y, ]


def generate(path):
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


if __name__ == '__main__':
    generate("strided_slice")