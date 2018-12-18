#!/usr/bin/env python

"""
:author Kier
"""


from node import Node

from binio import read_int
from binio import read_int_list
from binio import write_int
from binio import write_int_list

from node import write_bubble
from node import read_bubble

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


def write_nodes(stream, nodes, base=0):
    # type: (file, list[Node], int) -> None
    # build node_index_map
    index = base
    node_index_map = {}
    for node in nodes:
        node_index_map[node] = index
        index += 1

    # write number of nodes and each nodes
    write_int(stream=stream, i=len(nodes))
    for node in nodes:
        write_bubble(stream=stream, node=node)
        write_int_list(stream=stream, a=[node_index_map[input] for input in node.inputs])


def read_nodes(stream, base=0):
    # type: (file, int) -> list[Node]
    # read number of nodes
    nodes = []
    list_of_inputs = []
    size = read_int(stream=stream)
    while size > 0:
        nodes.append(read_bubble(stream=stream))
        list_of_inputs.append(read_int_list(stream=stream))
        size -= 1

    for i in range(len(nodes)):
        node = nodes[i]
        inputs = [nodes[j - base] for j in list_of_inputs[i]]
        Node.Link(node=node, inputs=inputs)

    return nodes
