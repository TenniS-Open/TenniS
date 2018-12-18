#!/usr/bin/env python

"""
:author Kier
"""


from node import Node
from graph import Graph

from tensor import compatible_string

import sys
if sys.version > '3':
    from queue import Queue
else:
    from Queue import Queue

class Module(object):
    def __init__(self):
        self.__inputs = []
        self.__outputs = []
        self.__nodes = []

    def load(self, nodes):
        # type: (Union[list[Node], Node]) -> None
        if isinstance(nodes, Node):
            nodes = [nodes, ]
        # walking on nodes
        inputs = []
        walker = Queue()
        walked = set()
        for node in nodes:
            walker.put(node)

        while not walker.empty():
            node = walker.get()
            if node in walked:
                continue
            walked.add(node)
            if node.op == node.Parameter:
                inputs.append(node)
                continue
            if node.op == node.Const:
                continue
            if len(node.inputs) == 0:
                raise Exception("Found not computable node: {}".format(node))
            for input in node.inputs:
                walker.put(input)

        input_set = set()
        for input in inputs:
            input_set.add(input)

        # save inputs and outputs
        self.__outputs.extend(nodes)
        self.__inputs.extend(list(input_set))
        # TODO: sort all nodes by depth
        self.__nodes.extend(list(walked))

    def clear(self):
        self.__inputs = []
        self.__outputs = []

    @property
    def inputs(self):
        # type: () -> list[Node]
        return self.__inputs

    @property
    def outputs(self):
        # type: () -> list[Node]
        return self.__outputs

    def sort_inputs(self, nodes):
        # type: (list[Union[Node, str]]) -> None
        # TODO: Check nodes is all inputs
        # build name node map
        name_node_set_map = {}
        for node in self.__inputs:
            name = node.name
            if name in name_node_set_map:
                name_node_set_map[name].append(node)
            else:
                name_node_set_map[name] = [node, ]
        # get inputs
        sorted_nodes = []
        for node in nodes:
            node = compatible_string(node)
            if isinstance(node, Node):
                pass
            elif isinstance(node, str):
                name = node
                if name in name_node_set_map:
                    node_set = name_node_set_map[name]
                    if len(node_set) > 1:
                        raise Exception('Found {} node with name={}'.format(len(node_set), name))
                    node = node_set[0]
                else:
                    raise Exception('Can not sort inputs with name={}'.format(name))
            else:
                raise Exception('Can not sort inputs with type='.format(type(node)))
            sorted_nodes.append(node)
        # check inputs
        for input in self.__inputs:
            if not input in sorted_nodes:
                raise Exception("The sorted inputs must content {}".format(input.name))

        self.__inputs = sorted_nodes

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
    a = Node(Node.Parameter, "a")
    b = Node(Node.Parameter, "b")
    c = Node(Node.Const, "c")
    d = Node("add", "d")
    Node.Link(d, (a, b, c))
    module = Module()
    module.load(d)
    module.sort_inputs(['b', 'a'])
    print(module.inputs)
    print(module.outputs)
    Module.Save(None, module=module)