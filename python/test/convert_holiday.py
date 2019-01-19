#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.holiday.converter import convert


def test():
    convert("test.ext.dat", "test.tsm", export_all=True, has_header=True)


if __name__ == '__main__':
    test()
