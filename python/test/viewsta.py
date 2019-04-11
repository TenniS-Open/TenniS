#!/usr/bin/env python
# coding: UTF-8

import sys
import os

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from tensorstack import orz
import json


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage viewsta.py filename.sta")
        exit()

    print(json.dumps(orz.sta2obj(sys.argv[1], binary_mode=2), indent=2))
