#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.dragon.converter import convert
from tensorstack.backend.api import *
import cv2
import math


def test():
    convert("model_dragon_frcnn.onnx", "frcnn_from_dragon.tsm")

if __name__ == '__main__':
    test()
