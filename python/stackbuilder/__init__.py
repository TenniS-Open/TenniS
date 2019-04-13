#!/usr/bin/env python

"""
:author Kier
"""

try:
    from . import holiday
except Exception as e:
    import sys
    sys.stderr.write("import holiday failed with: {}\n".format(e.message))
try:
    from . import onnx
except Exception as e:
    import sys
    sys.stderr.write("import onnx failed with: {}\n".format(e.message))
try:
    from . import torch
except Exception as e:
    import sys
    sys.stderr.write("import torch failed with: {}\n".format(e.message))
try:
    from . import vvvv
except Exception as e:
    import sys
    sys.stderr.write("import vvvv failed with: {}\n".format(e.message))
try:
    from . import caffe
except Exception as e:
    import sys
    sys.stderr.write("import caffe failed with: {}\n".format(e.message))
try:
    from . import mxnet
except Exception as e:
    import sys
    sys.stderr.write("import mxnet failed with: {}\n".format(e.message))
try:
    from . import tf
except Exception as e:
    import sys
    sys.stderr.write("import tensorflow failed with: {}\n".format(e.message))

__version__ = "0.0.9"
