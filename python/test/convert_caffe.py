#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.caffe.converter import convert


def test():
    root = "/Users/seetadev/Documents/Files/models/caffe/"
    prototxt = root + "deploy_x30.prototxt.txt"
    caffemodel = root + "pascalvoc2012_train_simple2_iter_30000.caffemodel"

    convert(prototxt, caffemodel, "caffe.tsm")


if __name__ == '__main__':
    test()
