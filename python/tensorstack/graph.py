#!/usr/bin/env python

"""
:author Kier
"""


from node import Node


class Graph(object):
    def __init__(self):
        self.__nodes = []

    def make(self, op=None, name=None, output_count=None):
        node = Node(op=op, name=name, output_count=output_count)
        self.__nodes.append(node)
        return node

    def clear(self):
        self.__nodes = []

    @property
    def nodes(self):
        return self.__nodes

