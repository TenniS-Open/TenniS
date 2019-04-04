#!python

import tensorflow as tf

if tf.__version__ < "1.10.0":
    raise ImportError("Please upgrade your tensorflow installation to v1.4.* or later! Got v{}.".format(tf.__version__))