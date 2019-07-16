#!python
# coding: UTF-8
"""
author: kier
"""

import sys

if sys.version_info.major == 2:
    import Queue as queue
else:
    import queue

import tensorstack as ts
import numpy

from .loadnet import Net
from .loadnet import hd
from .loadnet import OPType


class Node(object):
    def __init__(self, main, nodes=None, bottoms=None, tops=None):
        self.main = main
        if nodes is None:
            nodes = [main]
        self.nodes = nodes
        if bottoms is None:
            bottoms = main.inputs
        if tops is None:
            tops = [main]
        self.bottoms = bottoms
        self.tops = tops

    @classmethod
    def convert_add_bias(cls, ts_node):
        # type: (ts.Node) -> Node
        pre_ts_node = ts_node.inputs[0]
        pre_sn_node = cls.Convert(pre_ts_node)
        supported_combine = {"conv2d", "inner_prod"}
        if pre_sn_node.main.op in supported_combine:
            main = pre_sn_node.main
            nodes = []
            nodes.extend(pre_sn_node.nodes)
            nodes.append(ts_node)
            bottoms = pre_sn_node.bottoms
            tops = [ts_node]
            return Node(main, nodes, bottoms, tops)
        else:
            print ("Not valid: {}".format(pre_ts_node))
            return Node(ts_node, nodes=[ts_node], bottoms=[ts_node.inputs[0]], tops=[ts_node])

    @classmethod
    def convert_copy(cls, ts_node):
        # type: (ts.Node) -> Node
        return Node(ts_node)

    @classmethod
    def convert_param(cls, ts_node):
        # type: (ts.Node) -> Node
        label = ts.menu.data("label", 0, device=ts.device.CPU)
        return Node(ts_node, nodes=[ts_node], bottoms=[], tops=[ts_node, label])

    @classmethod
    def convert_flatten(cls, ts_node):
        # type: (ts.Node) -> Node
        node = ts_node.inputs[0]
        if node.op != "shape_index_patch":
            raise NotImplementedError("Only support flatten after shape_index_patch, got: {}".format(ts_node))
        return Node(node, nodes=[node, ts_node], bottoms=node.inputs, tops=[ts_node])

    @classmethod
    def convert_reshape(cls, ts_node):
        # type: (ts.Node) -> Node
        node = ts_node.inputs[0]
        support_nodes = {"inner_prod"}
        if node.op not in support_nodes:
            return Node(ts_node)
        pre_node = node.inputs[0]
        if pre_node.op == "flatten":
            return Node(node, [pre_node, node, ts_node],
                        bottoms=pre_node.inputs,
                        tops=ts_node)
        else:
            return Node(node, [node, ts_node],
                        bottoms=node.inputs,
                        tops=ts_node)

    @classmethod
    def convert_pooling2d(cls, ts_node):
        # type: (ts.Node) -> Node
        node = ts_node.inputs[0]
        if node.op != "pad":
            return Node(ts_node, [ts_node], bottoms=[ts_node.inputs[0]], tops=[ts_node])
        return Node(ts_node, [node, ts_node], bottoms=[node.inputs[0]], tops=[ts_node])

    @classmethod
    def convert_conv2d(cls, ts_node):
        # type: (ts.Node) -> Node
        node = ts_node.inputs[0]
        if node.op != "pad":
            return Node(ts_node, [ts_node], bottoms=[ts_node.inputs[0]], tops=[ts_node])
        return Node(ts_node, [node, ts_node], bottoms=[node.inputs[0]], tops=[ts_node])

    @classmethod
    def Convert(cls, ts_node):
        # type: (ts.Node) -> Node
        map_converter = {
            "add": cls.convert_copy,
            "to_float": cls.convert_copy,
            "relu": cls.convert_copy,
            "pooling2d": cls.convert_pooling2d,
            "conv2d": cls.convert_conv2d,
            "add_bias": cls.convert_add_bias,
            "_reshape": cls.convert_reshape,
            "flatten": cls.convert_flatten,
            "<param>": cls.convert_param,
        }
        if ts_node.op not in map_converter:
            raise NotImplementedError("{}".format(ts_node))
        converter = map_converter[ts_node.op]
        return converter(ts_node)


ts_node_to_sn_type = {
    "inner_prod": OPType.Enum_InnerProductLayer,
    "<param>": OPType.Enum_MemoryDataLayer,
    "to_float": OPType.Enum_SplitLayer,
    "conv2d": OPType.Enum_ConvolutionLayer,
    "relu": OPType.Enum_ReLULayer,
    "pooling2d": OPType.Enum_PoolingLayer,
    "shape_index_patch": OPType.Enum_ShapeIndexPatchLayer,
    "add": OPType.Enum_EltwiseLayer,
}

ts_node_to_converter = {
}


def register_converter(name, converter):
    ts_node_to_converter[name] = converter


class Graph(object):
    def __init__(self, nodes):
        self.nodes = nodes

    def save(self, path):
        # build seetanet
        layers = []
        blob_names = []
        layer_names = []
        last_layer = None
        blob2index = {}

        def add_blob_name(name):
            if name in blob_names:
                pass
            else:
                index = len(blob_names)
                blob_names.append(name)
                blob2index[name] = index

        for node in self.nodes:
            for top in node.tops:
                name = top.name
                add_blob_name(name)
            layer_names.append(node.tops[0].name)

        def create_layer(node, i, type):
            layer = hd.Holiday_LayerParameter()
            if node is not None:
                # layer.bottom = []
                # layer.bottom_index = []
                for bottom in node.bottoms:
                    layer.bottom.append(bottom.name)
                    layer.bottom_index.append(blob2index[bottom.name])
                # layer.top = []
                # layer.top_index = []
                for top in node.tops:
                    layer.top.append(top.name)
                    layer.top_index.append(blob2index[top.name])
                layer.name = node.main.name
            layer.type = type
            layer.layer_index = i
            return layer

        last_layer = create_layer(None, len(self.nodes), OPType.Enum_FinallyLayer)
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if node.main.op not in ts_node_to_sn_type or node.main.op not in ts_node_to_converter:
                raise NotImplementedError("Unsupported holiday layer: {}".format(node.main))
            layer_type = ts_node_to_sn_type[node.main.op]
            layer = create_layer(node, i, layer_type)
            converter = ts_node_to_converter[node.main.op]
            converter(node, layer)
            layers.append(layer)

        net = Net()
        net.layers = layers
        net.blob_names = blob_names
        net.layer_names = layer_names
        net.last_layer = last_layer

        with open(path, "wb") as f:
            net.save(f)

    @classmethod
    def _load(cls, ts_node):
        # type: (ts.Node) -> (Node, List[ts.Node])
        """
        :param ts_node:
        :return: sn_node, List[ts_node]
        """
        # print(ts_node.op)
        sn_node = Node.Convert(ts_node)
        # print(sn_node.main)
        return sn_node, sn_node.bottoms

    @classmethod
    def Load(cls, outputs):
        # type: (List[ts.Node]) -> Graph
        """
        :param outputs:
        :return:
        """
        que = queue.Queue()
        for output in outputs:
            que.put(output)

        # node2bottoms, for each node bottoms
        node2bottoms = {}
        # node2bottoms, for each node tops
        node2tops = {}
        # graph_nodes
        graph_nodes = []
        walked = set()

        while not que.empty():
            ts_node = que.get()
            sn_node, ts_inputs = cls._load(ts_node)
            for ts_input in ts_inputs:
                if ts_input not in walked:
                    que.put(ts_input)
            graph_nodes.append(sn_node)
            walked.add(ts_node)

        sorted_nodes = []

        # sort graph_nodes
        ready_bottoms = set()
        while len(graph_nodes) > 0:
            updated = False
            for i in range(len(graph_nodes)):
                node = graph_nodes[i]
                satisfied = True
                for bottom in node.bottoms:
                    if bottom.name not in ready_bottoms:
                        satisfied = False
                        break
                if satisfied:
                    sorted_nodes.append(node)
                    for top in node.tops:
                        ready_bottoms.add(top.name)
                    del graph_nodes[i]
                    updated = True
                    break
            if not updated:
                raise Exception("Unsatisfied bottoms of {}".format(graph_nodes))

        for node in sorted_nodes:
            print(node.main)

        return Graph(sorted_nodes)


def convert_sn_relu(node, layer):
    # type: (Node, hd.Holiday_LayerParameter) -> None
    pass


register_converter("relu", convert_sn_relu)


def convert_sn_param(node, layer):
    # type: (Node, hd.Holiday_LayerParameter) -> None
    param = layer.memory_data_param
    input_shape = node.main.get("#shape")
    param.batch_size = input_shape[0]
    param.channels = input_shape[1]
    param.height = input_shape[2]
    param.width = input_shape[3]


register_converter("<param>", convert_sn_param)


def convert_sn_split(node, layer):
    # type: (Node, hd.Holiday_LayerParameter) -> None
    pass


register_converter("to_float", convert_sn_split)


def dump_blob(blob, array, shape=None):
    if blob is None:
        blob = hd.Holiday_BlobProto()
    array = numpy.asarray(array, dtype=numpy.float32)
    if shape is None:
        shape = array.shape
    # blob.shape = hd.Holiday_BlobShape()
    for i in shape:
        blob.shape.dim.append(int(i))
    # blob.data = []
    for f in array.reshape([-1]):
        blob.data.append(f)
    return blob


def convert_sn_convolution(node, layer):
    # type: (Node, hd.Holiday_LayerParameter) -> None
    pad = None
    bias = None
    for item in node.nodes:
        if item.op == "pad":
            pad = item
        if item.op == "add_bias":
            bias = item

    weights = node.main.inputs[1]
    assert isinstance(weights, ts.Node)
    weights = weights.get("value")
    weights = numpy.asarray(weights)

    if bias is not None:
        bias = bias.inputs[1]
        bias = bias.get("value")

    param = layer.convolution_param

    dump_blob(param.kernel_param, weights)
    if bias is not None:
        dump_blob(param.bias_param, bias)

    dilation = node.main.get("dilation")
    stride = node.main.get("stride")
    padding = node.main.get("padding")

    if pad is not None:
        padding = numpy.asarray(padding, dtype=numpy.int32)
        padding += pad.inputs[1].get("value")

    param.dilation_height = dilation[2]
    param.dilation_width = dilation[3]
    param.num_output = weights.shape[0]
    param.pad_height = padding[2][0] + padding[2][1] // 2
    param.pad_width = padding[3][0] + padding[3][1] // 2
    param.kernel_height = weights.shape[2]
    param.kernel_width = weights.shape[3]
    param.stride_height = stride[2]
    param.stride_width = stride[3]
    param.group = 1


register_converter("conv2d", convert_sn_convolution)


def convert_sn_pooling(node, layer):
    # type: (Node, hd.Holiday_LayerParameter) -> None
    pad = None
    for item in node.nodes:
        if item.op == "pad":
            pad = item

    param = layer.pooling_param

    ksize = node.main.get("ksize")
    stride = node.main.get("stride")
    padding = node.main.get("padding")

    if pad is not None:
        padding = numpy.asarray(padding, dtype=numpy.int32)
        padding += pad.inputs[1].get("value")

    param.kernel_height = ksize[2]
    param.kernel_width = ksize[3]
    param.pad_height = padding[2][0] + padding[2][1] // 2
    param.pad_width = padding[3][0] + padding[3][1] // 2
    param.stride_height = stride[2]
    param.stride_width = stride[3]

    pooling_type = node.main.get("type")
    if pooling_type == 0:
        param.pool = hd.Holiday_PoolingParameter.MAX
    elif pooling_type == 1:
        param.pool = hd.Holiday_PoolingParameter.AVG
    else:
        raise NotImplementedError("Not support pooling type = {}".format(pooling_type))


register_converter("pooling2d", convert_sn_pooling)


def convert_sn_inner_prod(node, layer):
    # type: (Node, hd.Holiday_LayerParameter) -> None
    bias = None
    for item in node.nodes:
        if item.op == "add_bias":
            bias = item

    weights = node.main.inputs[1]
    assert isinstance(weights, ts.Node)
    weights = weights.get("value")
    weights = numpy.asarray(weights)

    if bias is not None:
        bias = bias.inputs[1]
        bias = bias.get("value")

    param = layer.inner_product_param

    if bias is not None:
        dump_blob(param.bias_param, bias)

    transpose = False
    if node.main.has("transpose"):
        transpose = bool(node.main.get("transpose"))

    param.num_output = weights.shape[0] if transpose else weights.shape[1]

    if transpose:
        weights = weights.transpose()

    param.transpose = True
    dump_blob(param.Inner_param, weights, (weights.shape[1], weights.shape[0]))


register_converter("inner_prod", convert_sn_inner_prod)


def convert_sn_shape_index_path(node, layer):
    # type: (Node, hd.Holiday_LayerParameter) -> None
    param = layer.shape_index_patch_param

    origin_patch = node.main.get("origin_patch")
    origin = node.main.get("origin")

    origin_patch = numpy.asarray(origin_patch)
    origin = numpy.asarray(origin)

    param.origin_patch.append(origin_patch[0])
    param.origin_patch.append(origin_patch[1])
    param.origin.append(origin[0])
    param.origin.append(origin[1])


register_converter("shape_index_patch", convert_sn_shape_index_path)


def convert_sn_add(node, layer):
    # type: (Node, hd.Holiday_LayerParameter) -> None
    param = layer.eltwise_param

    param.operation = hd.Holiday_EltwiseParameter.SUM


register_converter("add", convert_sn_add)
