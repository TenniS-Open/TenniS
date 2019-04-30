#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.caffe.converter import convert


def test():
    # root = "/Users/seetadev/Documents/Files/models/caffe/"
    # prototxt = root + "deploy_x30.prototxt.txt"
    # caffemodel = root + "pascalvoc2012_train_simple2_iter_30000.caffemodel"

    # root = "/Users/seetadev/Documents/Files/models/caffe/caffenet/"
    # prototxt = root + "caffenet.deploy.prototxt"
    # # prototxt = root + "caffenet.train.prototxt"
    # caffemodel = root + "bvlc_reference_caffenet.caffemodel"

    # root = "/Users/seetadev/Documents/Files/models/caffe/alexnet/"
    # # prototxt = root + "alexnet.deploy.prototxt"
    # prototxt = root + "alexnet.train_val.prototxt"
    # caffemodel = root + "bvlc_alexnet.caffemodel"

    root = "/Users/seetadev/Documents/Files/models/caffe/_TCID_model/"
    prototxt = root + "caffe.prototxt"
    caffemodel = root + "TCID8w-x9-128-c1.25.caffemodel"
    convert(prototxt, caffemodel, "caffe.tsm")

    #


if __name__ == '__main__':
    test()
