#!python
# coding: UTF-8
"""
author: kier
"""

from .proto import caffe_pb2 as caffe
from . import parser

import tensorstack as ts

from tensorstack import orz

from collections import OrderedDict
import numpy


def blob2numpy(blob):
    # type: (caffe.BlobProto) -> numpy.ndarray
    data = numpy.asarray(blob.data, dtype=numpy.float32)
    if len(data) == 0:
        return data
    shape = [-1, ]
    if blob.HasField("shape"):
        shape = tuple(blob.shape.dim)
    else:
        shape = (blob.num, blob.channels, blob.height, blob.width)
    return data.reshape(shape)


def convert(prototxt, caffemodel, output_file,
            input_layer_names=None, output_blob_names=None,
            include=None, exclude=None):
    """
    :param prototxt: path to prototxt file
    :param caffemodel: path to caffemodel file
    :param input_layer_names: list of input layers
    :param output_blob_names: list of output blobs
    :param output_file: path of output file
    :param include: list of string(phase), means using in net
    :param exclude: list of string(phase), means not using in net
    :return: None
    """
    net = caffe.NetParameter()
    with open(prototxt, "rb") as file_prototxt:
        from google.protobuf import text_format
        binary = file_prototxt.read()
        text = binary.decode("utf-8")
        text_format.Parse(text, net)
    params = caffe.NetParameter()
    with open(caffemodel, "rb") as file_caffemodel:
        binary = file_caffemodel.read()
        params.ParseFromString(binary)

    layer2params = {}
    # load params
    for param_layer in parser.include(params.layers, caffe.TEST):   # for V1LayerParameter
        layer_params = param_layer.blobs
        # print "{}: {}".format(param_layer.name, len(layer_params))
        layer2params[param_layer.name] = layer_params
    for param_layer in parser.include(params.layer, caffe.TEST):
        layer_params = param_layer.blobs
        # print "{}: {}".format(param_layer.name, len(layer_params))
        layer2params[param_layer.name] = layer_params

    # load net structure
    layers = parser.include(net.layer, caffe.TEST)

    if len(layers) == 0:
        raise Exception("Only support LayerParameter, not V0 or V1")

    input_name_node_map = OrderedDict()
    # output_name_node_map = OrderedDict()

    deploy_input_name = []
    deploy_input_shape = []

    if len(net.input) > 0 and len(net.input_shape) > 0:
        assert len(net.input) == len(net.input_shape)
        for i in range(len(net.input)):
            input_name = net.input[i]
            input_shape = parser.blob_shape(net.input_shape[i])
            deploy_input_name.append(input_name)
            deploy_input_shape.append(input_shape)

    def may_input_layer(converter):
        return orz.bind(converter,
                        orz.Placeholder(0), orz.Placeholder(1),
                        orz.Placeholder(2), orz.Placeholder(3),
                        input_name_node_map)

    # function format(layer, params, input_nodes, output_names, input_name_node_map=None)
    layer_converters = {
        # add layer converter here
        "ReLU": convert_relu_layer,
        "ImageData": may_input_layer(convert_image_data_layer),
        "Convolution": convert_convolution_layer,
        "BatchNorm": convert_batch_norm,
        "Scale": convert_scale,
        "Pooling": convert_pooling,
        "InnerProduct": convert_inner_product,
    }

    blob2nodes = {}
    # save blob used in top
    blob2count = {}

    def add_blob_count(blob):
        if blob in blob2count:
            blob2count[blob] += 1
        else:
            blob2count[blob] = 1

    # calculate each top count
    for input_name in deploy_input_name:
        add_blob_count(input_name)
    # ------------------------
    for layer in layers:
        # layer_params = layer.blobs
        # print "{}: {}".format(layer.name, len(layer_params))
        for top in layer.top:
            add_blob_count(top)

    # for input layer
    for i in range(len(deploy_input_name)):
        top = deploy_input_name[i]
        shape = deploy_input_shape[i]
        node_name = None
        if blob2count[top] <= 1:
            node_name = top
        else:
            blob2count[top] -= 1
            node_name = "{}_hide_{}".format(top, blob2count[top])
        input_node = ts.menu.param("_origin_" + node_name, shape)
        node = ts.zoo.to_float(node_name, input_node)
        blob2nodes[top] = node

        # collect inputs
        input_layer_names[node_name] = input_node

    # for each layer
    for layer in layers:
        # convert layer
        if layer.type not in layer_converters:
            raise Exception("Not supported Layer: {}".format(layer.type))
        ts_converter = layer_converters[layer.type]

        # gather input nodes
        input_nodes = []
        for bottom in layer.bottom:
            if bottom not in blob2nodes:
                raise Exception("Not computed blob: {}".format(bottom))
            input_nodes.append(blob2nodes[bottom])

        # set output names
        output_names = []
        for top in layer.top:
            if blob2count[top] <= 1:
                output_names.append(top)
            else:
                blob2count[top] -= 1
                output_names.append("{}_hide_{}".format(top, blob2count[top]))

        # query params
        params = []
        if layer.name in layer2params:
            params = layer2params[layer.name]

        ts_nodes = ts_converter(layer, params, input_nodes, output_names)

        if not isinstance(ts_nodes, list) and not isinstance(ts_nodes, tuple):
            ts_nodes = (ts_nodes, )

        assert len(ts_nodes) == len(layer.top)

        for i in range(len(layer.top)):
            # update blob2nodes
            blob2nodes[layer.top[i]] = ts_nodes[i]

    inputs = None
    if input_layer_names is not None:
        inputs = []
        for input_layer_name in input_layer_names:
            if input_layer_name not in input_name_node_map:
                raise Exception("There is no input named: {}".format(input_layer_name))
            inputs.append(input_name_node_map[input_layer_name])

    if output_blob_names is None:
        output_blob_names = []
        # count bottom
        bottom_set = set()
        for layer in layers:
            if len(layer.bottom) == 1 and len(layer.top) == 1 and layer.top[0] == layer.bottom[0]:
                continue
            for bottom in layer.bottom:
                bottom_set.add(bottom)
            if layer.type[-4:] == "Data":   # discard data label
                for top in layer.top[1:]:
                    bottom_set.add(top)
        top_set = set(blob2count.keys())
        output_blob_names = list(top_set - bottom_set)

    outputs = []
    for output_blob_name in output_blob_names:
        if output_blob_name not in blob2nodes:
            raise Exception("There is no blob named: {}".format(output_blob_name))
        outputs.append(blob2nodes[output_blob_name])

    module = ts.Module()

    # load module
    module.load(outputs)

    # sort inputs
    assert len(module.inputs) == 1

    with open(output_file, "wb") as fo:
        ts.Module.Save(stream=fo, module=module)

    print("============ Summary ============")
    # print("Input file: {}".format(input_file))
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


def convert_relu_layer(layer, params, input_nodes, output_names):
    # type: (caffe.LayerParameter, List[ts.Node], List[caffe.BlobProto], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer({}): {} ]=-".format(layer.type, layer.name, output_names))

    assert len(input_nodes) == 1
    assert len(params) == 0
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    node = ts.zoo.relu(node_name, x=x)

    return node,


def apply_transform(transform_param, node, suffix, log=True):
    # type: (caffe.TransformationParameter, ts.Node, str, bool) -> ts.Node
    if transform_param.HasField("crop_size"):
        crop_size = transform_param.crop_size
        print("--##    crop_size: {}".format(crop_size))
        node = ts.zoo.crop_nd("_crop_" + suffix, node, size=[-1, -1, crop_size, crop_size])

    mean_value = transform_param.mean_value
    if len(mean_value) > 0:
        print("--##    mean_value: {}".format(mean_value))
        rhs = numpy.asarray(mean_value, numpy.float32)
        rhs = numpy.reshape(rhs, newshape=[1, len(mean_value), 1, 1])
        node = ts.zoo.sub("_sub_mean_" + suffix, node, rhs)

    if transform_param.HasField("scale"):
        scale = transform_param.scale
        print("--##    scale: {}".format(scale))
        node = ts.zoo.mul("_scale_" + suffix, node, float(scale))

    return node


def convert_image_data_layer(layer, params, input_nodes, output_names, input_name_node_map):
    # type: (caffe.LayerParameter, List[ts.Node], List[caffe.BlobProto], List[str], Map[str, ts.Node]) -> List[ts.Node]
    print("--# -=[ Converting {} layer({}): {} ]=-".format(layer.type, layer.name, output_names))

    assert len(input_nodes) == 0
    assert len(output_names) == 2

    node_name = output_names[0]
    input_node = ts.menu.param("_origin_" + node_name)
    node = input_node

    layer_param = layer.image_data_param
    new_width = 0
    new_height = 0
    if layer_param.HasField("new_width"):
        print("--##    new_width: {}".format(new_width))
        new_width = layer_param.new_width
    if layer_param.HasField("new_height"):
        print("--##    new_height: {}".format(new_height))
        new_height = layer_param.new_height

    if new_width > 0 and new_height > 0:
        node = ts.zoo.resize2d("_resize2d_" + node_name, node, [-1, new_height, new_width, -1])

    node = ts.zoo.transpose("_nchw_" + node_name, node, pemute=[0, 3, 1, 2])
    node = ts.zoo.to_float("_float_" + node_name, node)

    # transform param
    if layer.HasField("transform_param"):
        node = apply_transform(layer.transform_param, node, node_name)

    node.name = node_name

    label_name = output_names[1]
    label = ts.menu.data(label_name, 1, ts.device.CPU)

    # set input node
    input_name_node_map[layer.name] = input_node

    return node, label


def message_getattr(message, attr, default):
    if message.HasField(attr):
        return getattr(message, attr)
    else:
        return default


def convert_convolution_layer(layer, params, input_nodes, output_names):
    # type: (caffe.LayerParameter, List[ts.Node], List[caffe.BlobProto], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer({}): {} ]=-".format(layer.type, layer.name, output_names))

    assert len(input_nodes) == 1
    assert len(params) > 0
    assert len(output_names) == 1

    conv2d_name = "_conv2d_" + output_names[0]
    bias_name = "_bias_" + output_names[0]
    node_name = output_names[0]

    layer_param = layer.convolution_param

    bias_term = True
    if layer_param.HasField("bias_term"):
        bias_term = layer_param.bias_term

    bias_blob = None    # [output_channels, ]
    if bias_term:
        assert len(params) > 1
        bias_blob = blob2numpy(params[1])
        print("--##    Bias shape: {}".format(bias_blob.shape))

    # is [output_channels, input_channels / group, input_height, input_width]
    weights_blob = blob2numpy(params[0])
    print("--##    Weights shape: {}".format(weights_blob.shape))

    force_nd_im2col = False
    if layer_param.HasField("force_nd_im2col"):
        force_nd_im2col = layer_param.force_nd_im2col
    if force_nd_im2col:
        raise NotImplementedError("force_nd_im2col = {}".format(force_nd_im2col))

    padding = [message_getattr(layer_param, "pad_h", 0),
               message_getattr(layer_param, "pad_w", 0)]

    if len(layer_param.pad) > 0:
        if len(layer_param.pad) == 1:
            padding = [layer_param.pad[0], layer_param.pad[0]]
        else:
            assert len(layer_param.pad) == 2
            padding = list(layer_param.pad)

    stride = [message_getattr(layer_param, "stride_h", 1),
              message_getattr(layer_param, "stride_w", 1)]

    if len(layer_param.stride) > 0:
        if len(layer_param.stride) == 1:
            stride = [layer_param.stride[0], layer_param.stride[0]]
        else:
            assert len(layer_param.stride) == 2
            stride = list(layer_param.stride)

    dilation = [1, 1]

    if len(layer_param.dilation) > 0:
        if len(layer_param.dilation) == 1:
            dilation = [layer_param.dilation[0], layer_param.dilation[0]]
        else:
            assert len(layer_param.dilation) == 2
            dilation = list(layer_param.dilation)

    print("--##    dilation: {}".format(dilation))
    print("--##    pad: {}".format(padding))
    print("--##    stride: {}".format(stride))

    num_output = None
    if layer_param.HasField("num_output"):
        num_output = layer_param.num_output

    kernel_size = [message_getattr(layer_param, "kernel_h", 0),
                   message_getattr(layer_param, "kernel_w", 0)]

    if len(layer_param.kernel_size) > 0:
        if len(layer_param.kernel_size) == 1:
            kernel_size = [layer_param.kernel_size[0], layer_param.kernel_size[0]]
        else:
            assert len(layer_param.kernel_size) == 2
            kernel_size = list(layer_param.kernel_size)

    print("--##    kernel_size: {}".format(kernel_size))

    assert kernel_size[0] == weights_blob.shape[2] and kernel_size[1] == weights_blob.shape[3]

    group = 1
    if layer_param.HasField("group"):
        group = layer_param.group
        print("--##    group: {}".format(group))

    input_channels = weights_blob.shape[1]

    assert weights_blob.shape[0] * group == num_output

    axis = 1
    if layer_param.HasField("axis"):
        axis = layer_param.axis

    assert axis == 1

    if group != 1 and weights_blob.shape[1] != 1:
        raise NotImplementedError("group = {} with weights.shape[1] = {}".format(group, weights_blob.shape[1]))

    is_conv2d = group == 1
    is_depthwise_conv2d = weights_blob.shape[1] == 1

    node = None

    if is_conv2d:
        node = ts.zoo.conv2d(conv2d_name, x=input_nodes[0], w=weights_blob, format=ts.zoo.Name.NCHW,
                             padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
                             padding_value=0,
                             stride=[1, 1, stride[0], stride[1]],
                             dilation=[1, 1, dilation[0], dilation[1]])
    elif is_depthwise_conv2d:
        weights_shape = weights_blob.shape
        depthwise_weights_shape = (weights_shape[1], weights_shape[0], weights_shape[2], weights_shape[3])
        weights_blob = weights_blob.reshape(shape=depthwise_weights_shape)
        node = ts.zoo.depthwise_conv2d(conv2d_name, x=input_nodes[0], w=weights_blob, format=ts.zoo.Name.NCHW,
                                       padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
                                       padding_value=0,
                                       stride=[0, 0, stride[0], stride[1]],
                                       dilation=[1, 1, dilation[0], dilation[1]])

    if node is None:
        raise NotImplementedError(layer)

    if bias_blob is not None:
        node = ts.zoo.add_bias(bias_name, x=node, b=bias_blob, format=ts.zoo.Name.NCHW)

    node.name = node_name

    return node,


def convert_batch_norm(layer, params, input_nodes, output_names):
    # type: (caffe.LayerParameter, List[ts.Node], List[caffe.BlobProto], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer({}): {} ]=-".format(layer.type, layer.name, output_names))

    assert len(input_nodes) == 1
    assert len(params) == 3
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    layer_param = layer.batch_norm_param

    mean = params[0]
    variance = params[1]

    mean = blob2numpy(mean)
    variance = blob2numpy(variance)

    scale_factor = blob2numpy(params[2])[0]

    scale = 0 if scale_factor == 0 else 1.0 / scale_factor

    mean *= scale
    variance *= scale

    mean = numpy.asarray(mean, numpy.float32)
    variance = numpy.asarray(variance, numpy.float32)

    print("--##    mean shape: {}".format(mean.shape))
    print("--##    variance shape: {}".format(variance.shape))
    print("--##    scale_factor: {}".format(scale_factor))

    epsilon = message_getattr(layer_param, "eps", 1e-5)

    node = ts.zoo.batch_norm(node_name, x=x, mean=mean, variance=variance, dim=1, epsilon=epsilon)

    return node,


def convert_scale(layer, params, input_nodes, output_names):
    # type: (caffe.LayerParameter, List[ts.Node], List[caffe.BlobProto], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer({}): {} ]=-".format(layer.type, layer.name, output_names))

    assert len(input_nodes) == 1 or len(input_nodes) == 2
    assert len(params) == 1 or len(params) == 2
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    layer_param = layer.scale_param

    scale = None
    if len(input_nodes) == 2:
        scale = input_nodes[1]
    else:
        scale = blob2numpy(params[0])
        print("--##    scale shape: {}".format(scale.shape))
    assert scale is not None

    bias_term = message_getattr(layer_param, "bias_term", False)

    bias = None
    if bias_term:
        assert len(params) == 2
        bias = blob2numpy(params[1])
        print("--##    bias shape: {}".format(bias.shape))

    axis = message_getattr(layer_param, "axis", 1)
    num_axes = message_getattr(layer_param, "num_axes", 1)

    print("--##    axis: {}".format(axis))
    print("--##    num_axes: {}".format(num_axes))

    node = None
    if axis == 1 and num_axes == 1:
        if bias is None:
            if not isinstance(scale, numpy.ndarray):
                raise NotImplementedError(layer)
            assert len(scale.shape) == 1
            scale = numpy.reshape(scale, newshape=[1, scale.shape[0], 1, 1])
            node = ts.zoo.mul(node_name, x, scale)
        else:
            node = ts.zoo.batch_scale(node_name, x, scale, bias, dim=1)
    else:
        print("WARNING: reach not fully supported setting.")
        if bias is None:
            node = ts.zoo.mul(node_name, x, scale)
        else:
            scale_node = ts.zoo.mul("_scale_" + node_name, x, scale)
            node = ts.zoo.add(node_name, scale_node, bias)

    if node is None:
        raise NotImplementedError(layer)

    return node,


def convert_pooling(layer, params, input_nodes, output_names):
    # type: (caffe.LayerParameter, List[ts.Node], List[caffe.BlobProto], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer({}): {} ]=-".format(layer.type, layer.name, output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    layer_param = layer.pooling_param

    pool = layer_param.pool

    caffe_pool_type_to_ts_pool_type = {
        layer_param.MAX: ts.zoo.Type.pooling_type.max,
        layer_param.AVE: ts.zoo.Type.pooling_type.avg,
        layer_param.STOCHASTIC: None,
    }

    caffe_pool_type_to_string = {
        layer_param.MAX: "MAX",
        layer_param.AVE: "AVE",
        layer_param.STOCHASTIC: "STOCHASTIC",
    }

    print("--##    Type : {}".format(caffe_pool_type_to_string[pool]))

    pooling_type = caffe_pool_type_to_ts_pool_type[pool]
    if pooling_type is None:
        raise NotImplementedError("Pooling type(code): {}({})".format(caffe_pool_type_to_string[pool], pool))

    global_pooling = False
    if layer_param.HasField("global_pooling"):
        global_pooling = layer_param.global_pooling
        print("--##    Global pooling: {}".format(global_pooling))

    if global_pooling:
        node = ts.zoo.global_pooling2d(node_name, x=x, type=pooling_type, format=ts.zoo.Name.NCHW)
        return node,

    padding = [message_getattr(layer_param, "pad_h", 0),
               message_getattr(layer_param, "pad_w", 0)]

    if layer_param.HasField("pad"):
        padding = [layer_param.pad, layer_param.pad]

    stride = [message_getattr(layer_param, "stride_h", 1),
              message_getattr(layer_param, "stride_w", 1)]

    if layer_param.HasField("stride"):
        stride = [layer_param.stride, layer_param.stride]

    kernel_size = [message_getattr(layer_param, "kernel_h", None),
                   message_getattr(layer_param, "kernel_w", None)]

    if layer_param.HasField("kernel_size"):
        kernel_size = [layer_param.kernel_size, layer_param.kernel_size]

    print("--##    pad: {}".format(padding))
    print("--##    stride: {}".format(stride))
    print("--##    kernel_size: {}".format(kernel_size))

    node = ts.zoo.pooling2d(node_name, x=x,
                            ksize=[1, 1, kernel_size[0], kernel_size[1]],
                            stride=[1, 1, stride[0], stride[1]],
                            type=pooling_type,
                            format=ts.zoo.Name.NCHW,
                            padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]])

    return node,


def convert_inner_product(layer, params, input_nodes, output_names):
    # type: (caffe.LayerParameter, List[ts.Node], List[caffe.BlobProto], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer({}): {} ]=-".format(layer.type, layer.name, output_names))

    assert len(input_nodes) == 1
    assert len(params) == 1 or len(params) == 2
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    layer_param = layer.inner_product_param

    num_output = None
    if layer_param.HasField("num_output"):
        num_output = layer_param.num_output

    axis = 1
    if layer_param.HasField("axis"):
        axis = layer_param.axis

    assert axis == 1

    transpose = False
    if layer_param.HasField("transpose"):
        transpose = layer_param.transpose
    print("--##    Transpose: {}".format(transpose))

    bias_term = True
    if layer_param.HasField("bias_term"):
        bias_term = layer_param.bias_term

    weights_blob = blob2numpy(params[0])
    print("--##    Weights shape: {}".format(weights_blob.shape))

    bias_blob = None
    if bias_term:
        assert len(params) == 2
        bias_blob = blob2numpy(params[1])
        print("--##    Bias shape: {}".format(bias_blob.shape))

    assert num_output == weights_blob.shape[0]
    num_input = weights_blob.shape[1]

    if transpose:
        weights_blob = numpy.reshape(weights_blob, newshape=[num_input, num_output])
    else:
        weights_blob = weights_blob.transpose()

    node = ts.zoo.flatten("_flatten_" + node_name, x=x)
    node = ts.zoo.inner_prod("_ip_" + node_name, node, weights_blob)

    if bias_blob is not None:
        node = ts.zoo.add_bias("_add_bias_" + node_name, x=node, b=bias_blob, format=ts.zoo.Name.NCHW)

    node = ts.zoo.reshape(name=node_name, x=node, shape=[-1, num_output, 1, 1])

    node.name = node_name

    return node,
