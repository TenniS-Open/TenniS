#!python
# coding: UTF-8
"""
author: kier
"""

try:
    from . import caffe_pb2
except:
    # compile proto
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    proto_root = path
    proto_file = os.path.join(proto_root, "caffe.proto")
    print("protoc {} --python_out={}".format(proto_file, proto_root))
    exit_status = os.system("protoc {} --proto_path={} --python_out={}".format(proto_file, proto_root, proto_root))
    if exit_status != 0:
        raise Exception("Can not compile proto: {}, please compile it by hand.".format(proto_file))
    from . import caffe_pb2
