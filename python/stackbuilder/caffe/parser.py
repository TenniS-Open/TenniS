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