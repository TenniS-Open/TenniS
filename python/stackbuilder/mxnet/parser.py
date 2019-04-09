#!python
# coding: UTF-8
"""
author: kier
"""


class Graph(object):
    def __init__(self, symbol, arg_params, aux_params=None):
        """
        :param symbol: json object
        :param arg_params: dict of arg_params
        :param aux_params: dict of aux_params
        """
        if aux_params is None:
            aux_params = {}
        self.__symbol = symbol
        self.__arg_params = arg_params
        self.__aux_params = aux_params
        self.__nodes = self.__symbol["nodes"]
        heads = self.__symbol["heads"]
        self.__heads = [head[0] for head in heads]
        self.__version = 0
        try:
            self.__version = symbol["attrs"]["mxnet_version"][1]
        except:
            pass

        for head in self.__heads:
            assert head >= 0 and head < len(self.__nodes)

    @property
    def symbol(self):
        return self.__symbol

    def param(self, name):
        if name in self.__arg_params:
            return self.__arg_params[name]
        if name in self.__aux_params:
            return self.__aux_params[name]
        return None

    @property
    def arg_params(self):
        return self.__arg_params

    @property
    def aux_params(self):
        return self.__aux_params

    @property
    def nodes(self):
        return self.__nodes

    @property
    def heads(self):
        return self.__heads

