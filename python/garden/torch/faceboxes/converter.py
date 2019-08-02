#!/usr/bin/env python

import stackbuilder as sb
import tensorstack as ts

from models.faceboxes import FaceBoxes
from models.faceboxes import CRelu
from models.faceboxes import Inception
from models.faceboxes import BasicConv2d

import torch
import numpy


def shape_hw(x, scope=None):
    if scope is None:
        scope = ""
    shape = ts.zoo.shape(name=scope + "/shape", x=x)
    hw = ts.frontend.onnx.gather(name=scope + "/hw", x=shape, indices=[2, 3])
    return hw


def shape_n(x, scope=None):
    if scope is None:
        scope = ""
    shape = ts.zoo.shape(name=scope + "/shape", x=x)
    n = ts.frontend.onnx.gather(name=scope + "/n", x=shape, indices=[0])
    return n


def convert_faceboxs(m, x, scope=None):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if scope is None:
        scope = ""

    assert isinstance(x, ts.Node)
    assert isinstance(m, FaceBoxes)
    """
    sources = list()
    loc = list()
    conf = list()
    detection_dimension = list()

    x = self.conv1(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = self.conv2(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.inception3(x)
    detection_dimension += [x.shape[2:]]
    sources.append(x)
    x = self.conv3_1(x)
    x = self.conv3_2(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    x = self.conv4_1(x)
    x = self.conv4_2(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    
    detection_dimension = torch.Tensor(detection_dimension)
    detection_dimension = detection_dimension.cuda()

    for (x, l, c) in zip(sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
    
    output = (loc.view(loc.size(0), -1, 4),
              self.softmax(conf.view(-1, self.num_classes)),
              detection_dimension)
  
    return output
    """

    sources = list()
    loc = list()
    conf = list()
    detection_dimension = list()

    self = m
    x = sb.torch.module.convert_module(self.conv1, x=x, scope=scope + "/conv1")
    x = ts.frontend.onnx.pooling2d(name=scope + "/pool1", x=x,
                                   ksize=3,
                                   stride=2,
                                   padding=1,
                                   type=ts.zoo.Type.pooling_type.max,
                                   padding_type=ts.zoo.Type.padding_type.white,
                                   auto_pad="NOTSET")
    x = sb.torch.module.convert_module(self.conv2, x=x, scope=scope + "/conv2")
    x = ts.frontend.onnx.pooling2d(name=scope + "/pool1", x=x,
                                   ksize=3,
                                   stride=2,
                                   padding=1,
                                   type=ts.zoo.Type.pooling_type.max,
                                   padding_type=ts.zoo.Type.padding_type.white,
                                   auto_pad="NOTSET")
    x = sb.torch.module.convert_module(self.inception1, x=x, scope=scope + "/inception1")
    x = sb.torch.module.convert_module(self.inception2, x=x, scope=scope + "/inception2")
    x = sb.torch.module.convert_module(self.inception3, x=x, scope=scope + "/inception3")

    detection_dimension.append(shape_hw(x, scope=scope + "/detection_dimension_0"))
    sources.append(x)

    x = sb.torch.module.convert_module(self.conv3_1, x=x, scope=scope + "/conv3_1")
    x = sb.torch.module.convert_module(self.conv3_2, x=x, scope=scope + "/conv3_2")

    detection_dimension.append(shape_hw(x, scope=scope + "/detection_dimension_1"))
    sources.append(x)

    x = sb.torch.module.convert_module(self.conv4_1, x=x, scope=scope + "/conv4_1")
    x = sb.torch.module.convert_module(self.conv4_2, x=x, scope=scope + "/conv4_2")

    detection_dimension.append(shape_hw(x, scope=scope + "/detection_dimension_2"))
    sources.append(x)

    detection_dimension = ts.frontend.tf.stack(name=scope + "/detection_dimension", tensors=detection_dimension)
    detection_dimension = ts.zoo.to_float(name=scope + "/detection_dimension_float", x=detection_dimension)

    i = 0
    for (x, l, c) in zip(sources, self.loc, self.conf):
        loc_i = sb.torch.module.convert_module(l, x=x, scope=scope + "/loc_%d" % i)
        loc_i = ts.zoo.transpose(name=scope + "/loc_%d_nhwc" % i, x=loc_i, permute=[0, 2, 3, 1])
        loc.append(loc_i)

        conf_i = sb.torch.module.convert_module(c, x=x, scope=scope + "/conf_%d" % i)
        conf_i = ts.zoo.transpose(name=scope + "/conf_%d_nhwc" % i, x=conf_i, permute=[0, 2, 3, 1])
        conf.append(conf_i)

        i += 1

    flatten_loc = []
    for l in loc:
        loc_i = ts.zoo.flatten(name=scope + "/loc_%d_flatten" % i, x=l)
        flatten_loc.append(loc_i)

    flatten_conf = []
    for c in conf:
        conf_i = ts.zoo.flatten(name=scope + "/conf_%d_flatten" % i, x=c)
        flatten_conf.append(conf_i)

    loc = ts.zoo.concat(name=scope + "/loc", inputs=flatten_loc, dim=1)
    conf = ts.zoo.concat(name=scope + "/conf", inputs=flatten_conf, dim=1)

    loc_shape_n = shape_n(x=loc, scope=scope + "/loc")
    loc_shape_tail = ts.menu.data(name=scope + "/loc/tail",
                                  value=numpy.asarray([-1, 4], numpy.int32), device=ts.device.CPU)

    loc_shape = ts.zoo.concat(name=scope + "/loc_shape", inputs=[loc_shape_n, loc_shape_tail], dim=0)
    loc_view = ts.zoo.reshape(name=scope + "/loc_view", x=loc, shape=loc_shape)

    conf_view = ts.zoo.reshape(name=scope + "/conf_view", x=conf, shape=[-1, self.num_classes])
    conf_softmax = ts.zoo.softmax(name=scope + "/conf_softmax", x=conf_view, dim=1)

    output = [loc_view, conf_softmax, detection_dimension]
    # output = ts.menu.pack(name=scope + "/output", inputs=output)

    return output


def convert_crelu(m, x, scope=None):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if scope is None:
        scope = ""

    assert isinstance(x, ts.Node)
    assert isinstance(m, CRelu)
    """
    x = self.conv(x)
    x = self.bn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x
    """

    self = m
    x = sb.torch.module.convert_module(self.conv, x=x, scope=scope + "/conv")
    x = sb.torch.module.convert_module(self.bn, x=x, scope=scope + "/bn")
    neg_x = ts.zoo.sub(name=scope + "/neg", lhs=numpy.asarray(0, dtype=numpy.float32), rhs=x)
    x = ts.zoo.concat(name=scope + "/cat", inputs=[x, neg_x], dim=1)
    x = ts.zoo.relu(name=scope + "/relu", x=x)

    return x


def convert_inception(m, x, scope=None):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if scope is None:
        scope = ""

    assert isinstance(x, ts.Node)
    assert isinstance(m, Inception)
    """
    branch1x1 = self.branch1x1(x)
    
    branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)
    
    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)
    
    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    branch3x3_3 = self.branch3x3_3(branch3x3_2)
    
    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return torch.cat(outputs, 1)
    """

    self = m

    branch1x1 = sb.torch.module.convert_module(self.branch1x1, x=x, scope=scope + "/branch1x1")
    branch1x1_pool = ts.frontend.onnx.pooling2d(name=scope + "/branch1x1_pool", x=x,
                                                ksize=3,
                                                stride=1,
                                                padding=1,
                                                type=ts.zoo.Type.pooling_type.avg,
                                                padding_type=ts.zoo.Type.padding_type.white,
                                                auto_pad="NOTSET")
    branch1x1_2 = sb.torch.module.convert_module(self.branch1x1_2, x=branch1x1_pool, scope=scope + "/branch1x1_2")

    branch3x3_reduce = sb.torch.module.convert_module(self.branch3x3_reduce, x=x, scope=scope + "/branch3x3_reduce")
    branch3x3 = sb.torch.module.convert_module(self.branch3x3, x=branch3x3_reduce, scope=scope + "/branch3x3")

    branch3x3_reduce_2 = sb.torch.module.convert_module(self.branch3x3_reduce_2, x=x,
                                                        scope=scope + "/branch3x3_reduce_2")
    branch3x3_2 = sb.torch.module.convert_module(self.branch3x3_2, x=branch3x3_reduce_2, scope=scope + "/branch3x3_2")
    branch3x3_3 = sb.torch.module.convert_module(self.branch3x3_3, x=branch3x3_2, scope=scope + "/branch3x3_3")

    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return ts.zoo.concat(name=scope + "/outputs", inputs=outputs, dim=1)


def convert_basic_conv2d(m, x, scope=None):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if scope is None:
        scope = ""

    assert isinstance(x, ts.Node)
    assert isinstance(m, BasicConv2d)
    """
    x = self.conv(x)
    x = self.bn(x)
    return F.relu(x, inplace=True)
    """
    self = m

    x = sb.torch.module.convert_module(self.conv, x=x, scope=scope + "/conv")
    x = sb.torch.module.convert_module(self.bn, x=x, scope=scope + "/bn")

    return ts.zoo.relu(name=scope + "/relu", x=x)


sb.torch.module.register_module_converter(FaceBoxes, convert_faceboxs)
sb.torch.module.register_module_converter(CRelu, convert_crelu)
sb.torch.module.register_module_converter(Inception, convert_inception)
sb.torch.module.register_module_converter(BasicConv2d, convert_basic_conv2d)


def convert_module(m, x=None):
    """
    convert module to graph node
    :param m:
    :param x:
    :return:
    """
    m.eval()
    with torch.no_grad():
        return sb.torch.module.convert_module(m, x)


def convert(input_module, output_file, input):
    """
    convert troch model to tsm
    :param input_module: torch.nn.Module or param can be parsed to troch.load(param)
    :param output_file: str of path to file
    :param input: list of tuple or ts.Node
    :return: ts.Module
    """

    return sb.torch.converter.convert(input_module=input_module, input=input, output_file=output_file)
