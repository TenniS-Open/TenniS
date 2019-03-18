import torch
# from torchsummary import summary

from . import module as convert_module

import numpy

from .onnx import convert as convert_onnx

import tempfile

from ..onnx import converter as onnx_converter


def convert(input_module, output_file, input):
    torch_model = None
    if isinstance(input_module, str):
        torch_model = torch.load(input_module)
    elif isinstance(input_module, torch.nn.Module):
        torch_model = input_module
    if torch_model is None:
        raise NotImplementedError("Not supported model: {}".format(type(input_module)))
    for param in torch_model.parameters():
        param.requires_grad = False
    torch_model.eval()

    temp_onnx_file = tempfile.mktemp()

    convert_onnx(torch_model, temp_onnx_file, input)

    onnx_converter.convert(temp_onnx_file, output_file)
