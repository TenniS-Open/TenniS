#!/usr/bin/env python

"""
:author Kier
"""

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

RuntimeRoot = "/home/kier/git/TensorStack/lib/"
sys.path.append(RuntimeRoot)

from tensorstack.backend.api import *

import cv2
import random


if __name__ == '__main__':
    device = Device("cpu", 0)

    cvimage = cv2.imread("/home/kier/000195.jpg")
    model = "/home/kier/yolov3.tsm"
    categories = [
        "bai_sui_shan",
        "cestbon",
        "cocacola",
        "jing_tian",
        "pepsi_cola",
        "sprite",
        "starbucks_black_tea",
        "starbucks_matcha",
        "starbucks_mocha",
        "vita_lemon_tea",
        "vita_soymilk_blue",
        "wanglaoji_green",
    ]

    bench = Workbench(device=device)
    bench.setup_context()
    bench.setup_device()
    bench.set_computing_thread_number(4)

    filter = ImageFilter(device=device)
    filter.channel_swap([2, 1, 0])
    filter.to_float()
    filter.scale(1 / 255.0)
    filter.letterbox(416, 416, 0.5)
    filter.to_chw()

    bench.setup(bench.compile(Module.Load(model)))
    cvimage_width = cvimage.shape[1]
    cvimage_height = cvimage.shape[0]

    tensor = Tensor(cvimage, UINT8, (1, cvimage.shape[0], cvimage.shape[1], cvimage.shape[2]))

    bench.bind_filter(0, filter)
    bench.input(0, tensor)

    bench.run()

    output_count = bench.output_count()

    for i in range(output_count):
        output = bench.output(i)
        output = output.cast(FLOAT32).numpy
        output_shape = output.shape
        N = output_shape[0]

        for n in range(N):
            x, y, w, h, score, label = output[n, :]
            label = int(label)
            x *= cvimage_width
            y *= cvimage_height
            w *= cvimage_width
            h *= cvimage_height

            R = random.Random(label)
            r = R.randint(64, 256)
            g = R.randint(64, 256)
            b = R.randint(64, 256)

            cv2.rectangle(cvimage, (int(x), int(y)), (int(x + w), int(y + h)), (b, g, r), 2)
            text = "{}: {}%".format(categories[label] if label < len(categories) else label, int(score * 100))

            print(text)

            cv2.putText(cvimage, text, (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (r - 32, g - 32, b - 32))

    cv2.imshow("Test", cvimage)
    cv2.waitKey()

    bench.dispose()
    filter.dispose()




