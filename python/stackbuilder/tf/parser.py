#!python
# coding: UTF-8
"""
author: kier
"""

import tensorflow as tf


def load_graph(path):
    """
    load graph from pb or tflite model
    :param path: path to pb file
    :return: tf.Graph
    """
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def load_ckpt(root, prefix=None):
    """
    load graph from ckpt files
    :param root: root to ckpt file and meta file
    :param prefix: prefix of meta file, exclude ".meta"
    :return: tf.Graph
    """
    if prefix is None:
        import os
        files = os.listdir(root)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % root)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % root)
        prefix = meta_files[0][:-5]

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}/{}.meta".format(root, prefix))
            saver.restore(sess, tf.train.latest_checkpoint(root))
    return graph
