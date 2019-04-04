#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.tf.converter import convert
import stackbuilder.tf.parser as parser

import tensorflow as tf


def test():
    graph = parser.loat_graph("tf/tensorflow.pb")
    print [n.name for n in graph.as_graph_def().node]

    with graph.as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            inputs = sess.graph.get_tensor_by_name("inputs_image:0")
            outputs = sess.graph.get_tensor_by_name("outputs:0")

            print inputs
            print outputs

            convert(graph,
                    inputs=inputs,
                    outputs=outputs,
                    output_file="tf.tsm")


if __name__ == '__main__':
    test()
