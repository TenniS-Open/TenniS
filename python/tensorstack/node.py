#!/usr/bin/env python

"""
:author Kier
"""


class Node(object):
    def __init__(self, op=None, name=None, output_count=None):
        self.__op = "" if op is None else op
        self.__name = "" if name is None else name
        self.__output_count = 0 if output_count is None else output_count
        self.__params = {
            "#op": self.__op,
            "#name": self.__name,
            "#output_count": self.__output_count,
        }
        self.__inputs = []
        self.__outputs = []

    @property
    def name(self):
        return self.__name

    @property
    def op(self):
        return self.__op

    @property
    def output_count(self):
        return self.__output_count

    @property
    def params(self):
        return self.__params

    def has(self, param):
        return param in self.__params

    def set(self, param, value):
        self.__params[param] = value

    def get(self, param):
        return self.__params[param]

    def clear(self, param):
        del self.__params[param]

    def clear_params(self):
        self.__params.clear()

    @property
    def inputs(self):
        return self.__inputs

    @property
    def outputs(self):
        return self.__outputs

    @staticmethod
    def link(node, inputs):
        """
        :param node: Node
        :param inputs: single Node of list of Node
        :return: None
        """
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            node.__inputs = list(inputs)
            for input in inputs:
                assert isinstance(input, Node)
                input.__outputs.append(node)
        elif isinstance(inputs, Node):
            input = inputs
            node.__inputs = [input]
            input.__outputs.append(node)
        else:
            raise Exception("Input nodes must be node or list of nodes")

