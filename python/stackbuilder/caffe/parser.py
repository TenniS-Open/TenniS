#!python
# coding: UTF-8
"""
author: kier
"""

from .proto import caffe_pb2 as caffe


def blob_shape(shape):
    # type: (caffe.BlobShape) -> List[int]
    assert isinstance(shape, caffe.BlobShape)
    return list(shape.dim)


def include(layers, phase):
    # type: (List[caffe.LayerParameter], caffe.Phase) -> List[caffe.LayerParameter]
    """
    remove all not in phase layer
    :param layers: list of caffe.LayerParameter
    :param phase: phase
    :return: list of layer in phase
    """
    include_layers = []
    for layer in layers:
        flag = False    # flag for continue
        for e in layer.exclude:
            if e.phase == phase:
                flag = True
                continue
        if flag:
            continue
        flag = len(layer.include) > 0
        for i in layer.include:
            if i.phase == phase:
                flag = False
                continue
        if flag:
            continue
        # print "name:", layer.name
        # print "include:", layer.include
        # print "exclude:", layer.exclude
        include_layers.append(layer)
    return include_layers