#!python

import torch
if torch.__version__ < "1.0.0":
    raise ImportError("Please upgrade your torch installation to v1.0.* or later! Got v{}.".format(torch.__version__))