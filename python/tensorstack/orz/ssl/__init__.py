#!/usr/bin/env python
# coding: UTF-8

try:
    from .aes import AESCrypto
except Exception, e:
    print("import AESCrypto failed: {}".format(e))
