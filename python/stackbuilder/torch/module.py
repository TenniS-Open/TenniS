#! python

"""
convert module
"""


import torch
import torchvision

import tensorstack as ts


def convert_conv2d(m, x):
    # type: (torch.nn.modules.conv.Conv2d, ts.Node) -> ts.Node
    print m
    for name, param in m.named_parameters():
        print name, param
    exit()


def convert_sequential(m, x):
    # type: (torch.nn.modules.container.Sequential, ts.Node) -> ts.Node
    for name, child in m.named_children():
        x = convert_module(child, x)
    return x


def convert_resnet(m, x):
    # type: (torchvision.models.resnet.ResNet, ts.Node) -> ts.Node
    return convert_sequential(m, x)


module2converter = {
    torchvision.models.resnet.ResNet: convert_resnet,
    torch.nn.modules.container.Sequential: convert_sequential,
    torch.nn.modules.conv.Conv2d: convert_conv2d,
}


def convert_module(m, x=None):
    # type: (torch.nn.Module, ts.Node) -> ts.Node
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

    return converter(m, x)



