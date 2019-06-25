# -*- coding: utf-8 -*-

"""
author: chao.yang
"""

from tensorstack.node import Node
import numpy as np

class QuantizeTranslatorOption:

    def __init__(self, quantizer_node_list):
        self.quantizer_node_list = quantizer_node_list

    def translate(self, compute_device, node):
        op_name = node.op

        if op_name != "conv2d" and op_name != "conv2d_v2" and op_name != "depthwise_conv2d" and op_name != "depthwise_conv2d_v2":
            return node
        
        if node.name not in self.quantizer_node_list.get().keys():
            return node

        format_tensor = node.get("format")
        stride_tenspr = node.get("stride")
        pad_tensor = node.get("padding")
        
        quantizer_node = self.quantizer_node_list.get()[node.name]
        quantize_scale = quantizer_node.bottom_scale
        weight_scale = quantizer_node.weight_scale

        dequantize_scales = np.zeros(np.size(weight_scale))
        for i in range(np.size(dequantize_scales)):
            if weight_scale[i] == 0 or quantize_scale == 0:
                dequantize_scales[i] = 0
            else:
                dequantize_scales[i] = 1 / (quantize_scale * weight_scale[i])
        
        inputs = node.inputs

        if op_name == "conv2d" or op_name == "depthwise_conv2d":
            kernel_node = inputs[1]
            kernel_quantize_name = kernel_node.name + "_quantize"
            kernel_quantize_node = Node("quantize",  kernel_quantize_name)
            kernel_quantize_node.set("quantize_scale", weight_scale)
            Node.Link(kernel_quantize_node, [kernel_node])

            quantize_name = inputs[0].name + "_quantize"
            quantize_node = Node("quantize", quantize_name)
            quantize_node.set("quantize_scale", quantize_scale)
            Node.Link(quantize_node, [inputs[0]])

            quantize_conv2d_name = node.name
            translated_node = Node("conv2d_quantized", quantize_conv2d_name)
            Node.Link(translated_node, [quantize_node, kernel_quantize_node])
        else:
            kernel_node = inputs[2]
            kernel_quantize_name = kernel_node.name + "_quantize"
            kernel_quantize_node = Node("quantize",  kernel_quantize_name)
            kernel_quantize_node.set("quantize_scale", weight_scale)
            Node.Link(kernel_quantize_node, [kernel_node])

            quantize_name = inputs[0].name + "_quantize"
            quantize_node = Node("quantize", quantize_name)
            quantize_node.set("quantize_scale", quantize_scale)
            Node.Link(quantize_node, [inputs[0]])

            quantize_conv2d_name = node.name
            translated_node = Node("conv2d_quantized", quantize_conv2d_name)
            Node.Link(translated_node, [quantize_node, inputs[1], kernel_quantize_node])

        translated_node.set("format", format_tensor)
        translated_node.set("stride", stride_tenspr)
        translated_node.set("padding", pad_tensor)
        if node.has("dilation"):
            translated_node.set("dilation", node.get("dilation"))
        if node.has("padding_value"):
            translated_node.set("padding_value", node.get("padding_value"))
        translated_node.set("dequantize_scales", dequantize_scales)

        return translated_node