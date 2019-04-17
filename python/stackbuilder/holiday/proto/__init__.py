#!/usr/bin/env python

"""
Author: Kier
"""

try:
    from . import HolidayCNN_proto_pb2
except:
    # compile proto
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    proto_root = path
    proto_file = os.path.join(proto_root, "HolidayCNN_proto.proto")
    print("protoc {} --python_out={}".format(proto_file, proto_root))
    exit_status = os.system("protoc {} --proto_path={} --python_out={}".format(proto_file, proto_root, proto_root))
    if exit_status != 0:
        raise Exception("Can not compile proto: {}, please compile it by hand.".format(proto_file))
    from . import HolidayCNN_proto_pb2

