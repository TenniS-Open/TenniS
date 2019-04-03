#!python

import mxnet

if mxnet.__version__ < "1.0.0":
    raise ImportError("Please upgrade your mxnet installation to v1.0.* or later! Got v{}.".format(mxnet.__version__))