#!/usr/bin/env python
# coding: UTF-8

try:
    from .aes import AESCrypto
except Exception as e:
    print("import AESCrypto failed: {}".format(e))
