#!python
# coding: UTF-8
"""
author: kier
"""

import tensorflow as tf


def version_satisfy(a, b):
    la = a.split('.')
    lb = b.split('.')
    f = min(len(la), len(lb))
    for i in range(f):
        try:
            if int(la[i]) > int(lb[i]):
                return True
            elif int(la[i]) == int(lb[i]):
                continue
            else:
                return False
        except IndexError as e:
            if len(la) > len(lb):
                return True
            else:
                return False
    return True

try:
    tf_version = tf.__version__
except:
    tf_version = "0.0.0"
if version_satisfy(tf_version, '1.14'):
    import_meta_graph = tf.compat.v1.train.import_meta_graph
    Session = tf.compat.v1.Session
else:
    import_meta_graph = tf.train.import_meta_graph
    Session = tf.Session


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


def session_load_graph_def(sess, path):
    """
    load graph from pb or tflite model
    :param sess: tf.Session
    :param path: path to pb file
    :return: tf.Graph
    """
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def session_load_ckpt(sess, root, prefix=None):
    """
    load graph from ckpt files
    :param sess: tf.Session
    :param root: root to ckpt file and meta file
    :param prefix: prefix of meta file, exclude ".meta"
    :return: None
    """
    import os, re
    if prefix is None:
        files = os.listdir(root)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % root)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % root)
        prefix = meta_files[0][:-5]

    meta_file = "{}.meta".format(prefix)
    ckpt_file = None

    ckpt = tf.train.get_checkpoint_state(root)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
    else:
        files = os.listdir(root)
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        if ckpt_file is None:
            raise ValueError('No ckpt file found in the model directory (%s)' % root)

    print("[INFO] import graph from: {}".format(os.path.join(root, meta_file)))
    saver = import_meta_graph(os.path.join(root, meta_file))
    print("[INFO] restore graph from: {}".format(os.path.join(root, ckpt_file)))
    saver.restore(sess, os.path.join(root, ckpt_file))
