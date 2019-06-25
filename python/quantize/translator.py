# -*- coding: utf-8 -*-

"""
author: chao.yang
"""

from tensorstack.module import Module
from tensorstack.node import Node

def translate_node(node, ready_map, compute_device, options):
    if node in ready_map:
        return ready_map[node]

    for option in options:
        translated_node = option.translate(compute_device, node)
    
    translated_inputs = []
    input_nodes = translated_node.inputs
    translated = False
    for input in input_nodes:
        translated_input = translate_node(input, ready_map, compute_device, options)
        if translated_input != input:
            translated = True
        translated_inputs.append(translated_input)
    
    ready_map[node] = translated_node

    if translated:
        Node.Link(translated_node, translated_inputs)
        
    return translated_node
    

class Translator:
    def __init__(self, compute_device, options):
        self.compute_device = compute_device
        self.options = options

    def translate(self, module):
        translated_nodes = []
        ready_map = {}

        output_nodes = module.outputs
        for node in output_nodes:
            translated_node = translate_node(node, ready_map, self.compute_device, self.options)
            translated_nodes.append(translated_node)

        new_module = Module()
        new_module.load(translated_nodes)

        return new_module