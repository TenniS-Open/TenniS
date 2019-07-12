# -*- coding: utf-8 -*-

"""
author: chao.yang
"""

from tensorstack.node import Node
import math
import numpy as np

def to_int8(input):
    temp = round(input)
    if temp > 127:
        return 127
    if temp < -128:
        return -128
    return temp

class QuantizeTranslatorOption:

    def __init__(self, quantizer_node_list):
        self.quantizer_node_list = quantizer_node_list

    def quantize_kernel(self, kernel_node, scale):
        kernel = kernel_node.get('value')
        kernel_int8_node = kernel_node
        count = kernel.size
        kernel_shape = kernel.shape
        # kernel_int8 = np.ndarray(kernel_shape, dtype=np.int8)
        quantize_group = scale.size
        loop_count = math.ceil(count/quantize_group)
        index = 0
        kernel_int8_list = []
        kernel_list = kernel.flatten()
        for n in range(quantize_group):
            quantize_scale = scale[n]
            loop_count_temp = loop_count
            while index < count and loop_count_temp > 0:
                int8_temp = to_int8(kernel_list[index] * quantize_scale)
                kernel_int8_list.append(int8_temp)
                index += 1
                loop_count_temp -= 1
        kernel_int8 = np.array(kernel_int8_list, dtype=np.int8).reshape(kernel_shape)
        kernel_int8_node.set('value', kernel_int8)
        return kernel_int8_node


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
            kernel_int8_node = self.quantize_kernel(kernel_node, weight_scale)
            # kernel_quantize_name = kernel_node.name + "_quantize"
            # kernel_quantize_node = Node("quantize",  kernel_quantize_name)
            # kernel_quantize_node.set("quantize_scale", weight_scale)
            # Node.Link(kernel_quantize_node, [kernel_node])

            quantize_name = inputs[0].name + "_quantize"
            quantize_node = Node("quantize", quantize_name)
            quantize_node.set("quantize_scale", quantize_scale)
            Node.Link(quantize_node, [inputs[0]])

            quantize_conv2d_name = node.name
            translated_node = Node("conv2d_quantized", quantize_conv2d_name)
            Node.Link(translated_node, [quantize_node, kernel_int8_node])
        else:
            kernel_node = inputs[2]
            kernel_int8_node = self.quantize_kernel(kernel_node, weight_scale)
            # kernel_quantize_name = kernel_node.name + "_quantize"
            # kernel_quantize_node = Node("quantize",  kernel_quantize_name)
            # kernel_quantize_node.set("quantize_scale", weight_scale)
            # Node.Link(kernel_quantize_node, [kernel_node])

            quantize_name = inputs[0].name + "_quantize"
            quantize_node = Node("quantize", quantize_name)
            quantize_node.set("quantize_scale", quantize_scale)
            Node.Link(quantize_node, [inputs[0]])

            quantize_conv2d_name = node.name
            translated_node = Node("conv2d_quantized", quantize_conv2d_name)
            Node.Link(translated_node, [quantize_node, inputs[1], kernel_int8_node])

        translated_node.set("format", format_tensor)
        translated_node.set("stride", stride_tenspr)
        translated_node.set("padding", pad_tensor)
        if node.has("dilation"):
            translated_node.set("dilation", node.get("dilation"))
        if node.has("padding_value"):
            translated_node.set("padding_value", node.get("padding_value"))
        translated_node.set("dequantize_scales", dequantize_scales)

        return translated_node