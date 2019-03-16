#!python

import onnx

if onnx.__version__ < "1.4.0":
    raise ImportError("Please upgrade your onnx installation to v1.4.* or later! Got v{}.".format(onnx.__version__))