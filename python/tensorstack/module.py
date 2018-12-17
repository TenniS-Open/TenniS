#!/usr/bin/env python

"""
:author Kier
"""


from node import Node
from graph import Graph


class Module(object):
    def __init__(self):
        self.__inputs = []
        self.__outputs = []
        pass

    def load(self, nodes):
        # type: (list[Node]) -> None
        pass

    def clear(self):
        pass

    @property
    def inputs(self):
        # type: () -> list[Node]
        pass

    @property
    def outputs(self):
        # type: () -> list[Node]
        pass

    def sort_inputs(self, nodes):
        # type: (list[Node]) -> None
        pass

    @staticmethod
    def Save(stream, module):
        # type: (file, Module) -> None
        module.__inputs = []
        pass

    @staticmethod
    def Load(stream):
        # type: (file) -> Module
        pass


if __name__ == '__main__':
    module = Module()
    Module.Save(None, module=module)