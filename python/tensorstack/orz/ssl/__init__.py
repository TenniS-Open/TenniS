#!/usr/bin/env python
# coding: UTF-8


class FakeAESCrypto(object):
    def __init__(self, key, iv=None):
        raise NotImplementedError("AESCrypto")


try:
    from .aes import AESCrypto
except Exception as e:
    # print("import AESCrypto failed: {}".format(e))
    AESCrypto = FakeAESCrypto
