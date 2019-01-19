#!/usr/bin/env python

"""
:author Kier
"""


from .node import Node
from .graph import Graph

from .tensor import compatible_string

from .binio import write_int
from .binio import write_int_list
from .binio import read_int_list
from .binio import read_int_list

from .header import Header
from .header import write_header
from .header import read_header

from .graph import write_nodes
from .graph import read_nodes

import io

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
        # type: (Union[file, io.BinaryIO, io.TextIO, io.StringIO], Module) -> None
        header = Header(code=Header.MODULE_CODE_V1)
        nodes = module.__nodes
        index = 0
        node_index_map = {}
        for node in nodes:
            node_index_map[node] = index
            index += 1
        inputs = [node_index_map[input] for input in module.__inputs]
        outputs = [node_index_map[output] for output in module.__outputs]
        # 0. write header
        write_header(stream=stream, header=header)
        # 1. write inputs
        write_int_list(stream=stream, a=inputs)
        # 2. write outputs
        write_int_list(stream=stream, a=outputs)
        # 3. write graph
        write_nodes(stream=stream, nodes=nodes)


    @staticmethod
    def Load(stream):
        # type: (Union[file, io.BinaryIO, io.TextIO, io.StringIO]) -> Module
        header = read_header(stream=stream)
        if header.code != Header.MODULE_CODE_V1:
            raise Exception("Do NOT support model format code={}".format(header.code))
        inputs = read_int_list(stream=stream)
        outputs = read_int_list(stream=stream)
        nodes = read_nodes(stream=stream)

        module = Module()
        module.__inputs = [nodes[i] for i in inputs]
        module.__outputs = [nodes[i] for i in outputs]
        module.__nodes = nodes

        return module


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

    with open("module.txt", "wb") as fo:
        Module.Save(fo, module=module)

    with open("module.txt", "rb") as fi:
        module = Module.Load(fi)

    with open("module.txt", "wb") as fo:
        Module.Save(fo, module=module)

    with open("module.txt", "rb") as fi:
        local_module = Module.Load(fi)
        print(local_module.inputs)
        print(local_module.outputs)
