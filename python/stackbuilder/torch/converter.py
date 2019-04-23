import torch
# from torchsummary import summary

from .module import convert_module

import numpy

import tensorstack as ts

# from .onnx import convert as convert_onnx
# from ..onnx import converter as onnx_converter


def convert(input_module, input, output_file):
    """
    convert troch model to tsm
    :param input_module: torch.nn.Module or param can be parsed to troch.load(param)
    :param input: list of tuple or ts.Node
    :param output_file: str of path to file
    :return: ts.Module
    """
    torch_model = None
    if isinstance(input_module, str):
        torch_model = torch.load(input_module)
    elif isinstance(input_module, torch.nn.Module):
        torch_model = input_module
    if torch_model is None:
        raise NotImplementedError("Not supported model: {}".format(type(input_module)))
    for param in torch_model.parameters():
        param.requires_grad = False

    if not isinstance(input, (tuple, list)):
        raise RuntimeError("input must be a list of tuple of ts.Node")

    input_nodes = []
    for i in range(len(input)):
        node = input[i]
        if isinstance(node, ts.Node):
            input_nodes.append(node)
        elif isinstance(node, (tuple, list)):
            for i in node:
                if not isinstance(i, int):
                    raise RuntimeError("input must be a list of tuple of ts.Node")
            input_nodes.append(ts.menu.param("_input_%d" % (i, ), shape=node))
        else:
            raise RuntimeError("input must be a list of tuple of ts.Node")

    assert isinstance(input_module, torch.nn.Module)

    module = None
    torch_model.eval()
    with torch.no_grad():
        ts_graph_outputs = convert_module(input_module, input_nodes)

        ts_module = ts.module.Module()
        ts_module.load(ts_graph_outputs)
        ts_module.sort_inputs(input_nodes)

        module = ts_module

    with open(output_file, "wb") as fo:
        ts.Module.Save(stream=fo, module=module)

    print("============ Summary ============")
    print("Input file: {}".format(input_nodes))
    print("Output file: {}".format(output_file))
    index = 0
    print("Input node: ")
    for node in module.inputs:
        assert isinstance(node, ts.Node)
        print("{}: {}, shape={}".format(index, node.name, node.shape))
        index += 1
    index = 0
    print("Output node: ")
    for node in module.outputs:
        assert isinstance(node, ts.Node)
        print("{}: {}".format(index, node.name))
        index += 1

    return module

