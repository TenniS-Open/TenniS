#! python

"""
convert module
"""


import torch
import torchvision

import tensorstack as ts
import numpy


def convert_conv2d(m, x, scope=None):
    # type: (torch.nn.modules.conv.Conv2d, ts.Node, str) -> ts.Node
    if isinstance(x, (tuple, list)):
        x = x[0]
    if scope is None:
        scope = ''
    assert isinstance(x, ts.Node)
    assert isinstance(m, torch.nn.modules.conv.Conv2d)

    print("--# -=[ Converting {} layer: {} ]=-".format(m.__class__.__name__, scope))
    print("--##    REPR: {}".format(m))
    """
    TODO: check if this shape OK
    output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
    """

    in_channels = m.in_channels
    out_channels = m.out_channels
    kernel_size = m.kernel_size
    stride = m.stride
    padding = m.padding
    dilation = m.dilation
    output_padding = m.output_padding
    groups = m.groups
    transposed = m.transposed
    """
    if transposed:
        self.weight = Parameter(torch.Tensor(
            in_channels, out_channels // groups, *kernel_size))
    else:
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
    """
    bias = m.bias
    """
    if bias:
        self.bias = Parameter(torch.Tensor(out_channels))
    else:
        self.register_parameter('bias', None)
    """
    weight = numpy.asarray(m.weight.cpu(), dtype=numpy.float32)
    bias = None

    if groups != 1:
        raise NotImplementedError("groups = {}".format(groups))

    if len(kernel_size) != 2:
        raise NotImplementedError("kernel_size = {}".format(kernel_size))

    if len(padding) != 2:
        raise NotImplementedError("padding = {}".format(padding))

    if output_padding != (0,) * len(output_padding):
        raise NotImplementedError("output_padding = {}".format(output_padding))

    if len(stride) != 2:
        raise NotImplementedError("stride = {}".format(stride))

    if len(dilation) != 2:
        raise NotImplementedError("stride = {}".format(dilation))

    if transposed:
        weight = weight.transpose(1, 0, 2, 3)

    assert in_channels == weight.shape[1] * groups
    assert out_channels == weight.shape[0]

    if m.bias is not None:
        bias = numpy.asarray(m.bias.cpu(), dtype=numpy.float32)

    conv2d_name = scope + "_conv"
    bias_name = scope + "_bias"

    x = ts.zoo.conv2d(name=conv2d_name, x=x, w=weight, format=ts.zoo.Name.NCHW,
                      padding=[(0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])],
                      stride=[1, 1, stride[0], stride[1]],
                      dilation=[1, 1, dilation[0], dilation[1]])

    if bias is not None:
        assert len(bias.shape) == 1
        x = ts.zoo.add_bias(name=bias_name, x=x, b=bias, dim=1)

    x.name = scope

    return x


def convert_sequential(m, x, scope=None):
    # type: (torch.nn.modules.container.Sequential, ts.Node, str) -> ts.Node
    if scope is None:
        scope = ''
    for name, child in m.named_children():
        x = convert_module(child, x, scope=scope + "/" + name)
    return x


def convert_resnet(m, x, scope=None):
    # type: (torchvision.models.resnet.ResNet, ts.Node, str) -> ts.Node
    return convert_sequential(m, x, scope=scope)


module2converter = {
    torchvision.models.resnet.ResNet: convert_resnet,
    torch.nn.modules.container.Sequential: convert_sequential,
    torch.nn.modules.conv.Conv2d: convert_conv2d,
}


def register_module_converter(module, converter):
    module2converter[module] = converter


def convert_module(m, x=None, scope=None):
    # type: (torch.nn.Module, Union[ts.Node, tuple[ts.Node], list[ts.Node]], str) -> ts.Node
    cls = type(m)

    converter = None
    for c in module2converter.keys():
        if cls == c:
            converter = module2converter[c]
            break

    if converter is None:
        raise NotImplementedError("Can not find any converter for {}".format(cls))

    if x is None:
        x = ts.menu.param("_input")

    if scope is None:
        scope = ''

    x = converter(m, x, scope=scope)

    return x
