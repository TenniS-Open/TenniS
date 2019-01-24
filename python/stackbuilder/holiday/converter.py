#!python
# coding: UTF-8
"""
author: kier
"""


from .loadnet import load_net
from .loadnet import hd
from .loadnet import OPType

import tensorstack as ts

import numpy


class HolidayNode(object):
    def __init__(self, layer, blob_names):
        # type: (hd.Holiday_LayerParameter, List[str]) -> None
        self.__tops = []
        self.__bottoms = []
        self.__layer = layer
        for top_index in self.__layer.top_index:
            self.__tops.append(blob_names[top_index])
        for bottom_index in self.__layer.bottom_index:
            self.__bottoms.append(blob_names[bottom_index])

    @property
    def tops(self):
        return self.__tops

    @property
    def bottoms(self):
        return self.__bottoms

    @property
    def layer(self):
        return self.__layer


def blob2numpy(blob):
    # type: (hd.Holiday_BlobProto) -> numpy.ndarray
    data = numpy.asarray(blob.data, dtype=float)
    if len(data) == 0:
        return data
    shape = tuple(blob.shape.dim)
    return data.reshape(shape)


def convert(input_file, output_file,
            output_blobs=None,
            has_header=False,
            export_all=False):
    header, net = load_net(input_file, has_header)

    if output_blobs is None:
        output_blobs = []

    if isinstance(output_blobs, tuple):
        output_blobs = list(output_blobs)
    elif not isinstance(output_blobs, list):
        output_blobs = [output_blobs, ]

    if export_all:
        output_blobs.extend(net.blob_names)

    if header is not None and header.blob_name not in output_blobs:
        output_blobs.append(header.blob_name)

    if len(output_blobs) == 0:
        raise Exception("#param output_blobs must be set as having no header")

    layer_converters = {
        OPType.Enum_MemoryDataLayer: convert_memorydata_layer,
        OPType.Enum_ConvolutionLayer: convert_convolution_layer,
        OPType.Enum_BatchNormliseLayer: convert_batch_norm_layer,
        OPType.Enum_ScaleLayer: convert_batch_scale_layer,
        OPType.Enum_ReLULayer: convert_relu_layer,
        OPType.Enum_PoolingLayer: convert_pooling_layer,
        OPType.Enum_InnerProductLayer: convert_inner_product_layer,
        OPType.Enum_EltwiseLayer: convert_eltwise_layer,
    }

    nodes = []
    blob2nodes = {}
    # save blob used in top
    blob2count = {}

    for layer in net.layers:
        assert isinstance(layer, hd.Holiday_LayerParameter)
        node = HolidayNode(layer=layer, blob_names=net.blob_names)
        nodes.append(node)

    for node in nodes:
        for top in node.tops:
            if top in blob2count:
                blob2count[top] += 1
            else:
                blob2count[top] = 1

    for node in nodes:
        layer = node.layer
        # convert layer
        if layer.type not in layer_converters:
            raise Exception("Not supported Layer(code): {}({})".format(OPType.EnumString[layer.type], layer.type))
        ts_converter = layer_converters[layer.type]

        input_nodes = []
        for bottom in node.bottoms:
            if bottom not in blob2nodes:
                raise Exception("Not computed blob: {}".format(bottom))
            input_nodes.append(blob2nodes[bottom])

        output_names = []
        for top in node.tops:
            if blob2count[top] <= 1:
                output_names.append(top)
            else:
                blob2count[top] -= 1
                output_names.append("{}_hide_{}".format(top, blob2count[top]))

        ts_nodes = ts_converter(layer=layer, input_nodes=input_nodes, output_names=output_names)

        assert len(ts_nodes) == len(node.tops)

        for i in range(len(node.tops)):
            # update blob2nodes
            blob2nodes[node.tops[i]] = ts_nodes[i]

    # get outputs from outout_blobs
    outputs = []
    for blob in output_blobs:
        if blob not in blob2nodes:
            raise Exception("Not computed blob: {}".format(blob))
        outputs.append(blob2nodes[blob])

    module = ts.Module()

    # load module
    module.load(outputs)

    # sort inputs
    assert len(module.inputs) == 1

    with open(output_file, "wb") as fo:
        ts.Module.Save(stream=fo, module=module)

    print("============ Summary ============")
    print("Input file: {}".format(input_file))
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


def convert_memorydata_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} ]=-".format(OPType.EnumString[layer.type], output_names))

    assert len(input_nodes) == 0
    assert len(output_names) == 2

    param = layer.memory_data_param

    input_batch_size = param.batch_size
    input_channels = param.channels
    input_height = param.height
    input_width = param.width

    if param.HasField("crop_size_height"):
        input_height = param.crop_size_height

    if param.HasField("crop_size_width"):
        input_width = param.crop_size_width

    input_shape = [input_batch_size, input_channels, input_height, input_width]

    print("--##    Input shape: {}".format(input_shape))

    input = ts.menu.param("_input", shape=input_shape)
    label = ts.menu.data(output_names[1], value=0, device=ts.device.CPU)

    input = ts.zoo.to_float("_float_input", x=input)

    scale = 1
    if param.HasField("scale"):
        scale = param.scale
        print("--##    Scale: {}".format(scale))

    mean_file = None
    if param.HasField("mean_file"):
        mean_file = param.mean_file
        print("--##    Mean file: {}".format(mean_file is not None))

    mean_value = list(param.mean_value)

    channel_waps = list(param.channel_swaps)
    if len(channel_waps) > 0:
        print("--##    Channel swap: {}".format(channel_waps))

    prewhiten = False
    if param.HasField("prewhiten"):
        prewhiten = param.prewhiten
        print("--##    Prewhiten: {}".format(prewhiten))

    if mean_file is not None:
        mean = blob2numpy(mean_file)
        mean = numpy.reshape(mean, newshape=[1, input_channels, input_height, input_width])
        input = ts.zoo.sub("_sub_mean_input", input, mean)
    elif len(mean_value) > 0:
        if len(mean_value) != input_channels:
            raise Exception("mean value size must be the input channels size")
        mean = numpy.asarray(mean_value, dtype=float).reshape(shape=[1, input_channels, 1, 1])
        input = ts.zoo.sub("_sub_mean_input", input, mean)

    if scale != 1:
        scale = numpy.asarray(scale, dtype=numpy.float32)
        input = ts.zoo.mul("_mul_scale_input", input, scale)

    if len(channel_waps) > 0:
        input = ts.zoo.dimsuffle("_channel_swap_input",
                                 input, dim=1, shuffle=numpy.asarray(channel_waps, dtype=numpy.int32))

    if prewhiten:
        input = ts.zoo.prewhiten("_prewhiten_input", input)

    input.name = output_names[0]

    return input, label


def convert_convolution_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} ]=-".format(OPType.EnumString[layer.type], output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    conv2d_name = output_names[0]

    param = layer.convolution_param

    bias_blob = None    # [output_channels, ]
    if param.HasField("bias_param"):
        bias_blob = blob2numpy(param.bias_param)
        print("--##    Bias shape: {}".format(bias_blob.shape))

    # is [output_channels, input_channels / group, input_height, input_width]
    weights_blob = blob2numpy(param.kernel_param)
    print("--##    Weights shape: {}".format(weights_blob.shape))

    dialation = [param.dilation_height, param.dilation_width]
    print("--##    Dialation: {}".format(dialation))

    num_output = None
    if param.HasField("num_output"):
        num_output = param.num_output

    padding = [param.pad_height, param.pad_width]
    print("--##    Padding: {}".format(padding))

    kernel_size = [param.kernel_height, param.kernel_width]

    assert kernel_size[0] == weights_blob.shape[2] and kernel_size[1] == weights_blob.shape[3]

    stride = [param.stride_height, param.stride_width]
    print("--##    Stride: {}".format(stride))

    group = 1
    if param.HasField("group"):
        group = param.group
        print("--##    Group: {}".format(group))

    input_channels = weights_blob.shape[1]

    assert weights_blob.shape[0] * group == num_output

    axis = 1
    if param.HasField("axis"):
        axis = param.axis

    assert axis == 1

    force_nd_im2col = False
    if param.HasField("force_nd_im2col"):
        force_nd_im2col = param.force_nd_im2col

    assert not force_nd_im2col

    tf_padding = None
    if param.HasField("tf_padding"):
        tf_padding = param.tf_padding
        print("--##    TF padding: {}".format(tf_padding))

    assert tf_padding is None or tf_padding == "SAME" or tf_padding == "VALID"

    if tf_padding is not None:
        raise NotImplementedError("TF padding = {}".format(tf_padding))

    if group != 1 and weights_blob.shape[1] != 1:
        raise NotImplementedError("group = {} with weights.shape[1] = {}".format(group, weights_blob.shape[1]))

    is_conv2d = group == 1
    is_depthwise_conv2d = weights_blob.shape[1] == 1

    conv2d = None

    if is_conv2d:
        conv2d = ts.zoo.conv2d(conv2d_name, x=input_nodes[0], w=weights_blob, format=ts.zoo.Name.NCHW,
                               padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
                               padding_value=0,
                               stride=[1, 1, stride[0], stride[1]],
                               dialations=[1, 1, dialation[0], dialation[1]])
    elif is_depthwise_conv2d:
        weights_shape = weights_blob.shape
        depthwise_weights_shape = (weights_shape[1], weights_shape[0], weights_shape[2], weights_shape[3])
        weights_blob = weights_blob.reshape(shape=depthwise_weights_shape)
        conv2d = ts.zoo.depthwise_conv2d(conv2d_name, x=input_nodes[0], w=weights_blob, format=ts.zoo.Name.NCHW,
                                         padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
                                         padding_value=0,
                                         stride=[0, 0, stride[0], stride[1]],
                                         dialations=[1, 1, dialation[0], dialation[1]])

    if conv2d is None:
        raise NotImplementedError(layer)

    return conv2d,


def convert_batch_norm_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} ]=-".format(OPType.EnumString[layer.type], output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    param = layer.batchNormlise_param

    mean = param.mean_param
    covariance = param.covariance_param

    mean = blob2numpy(mean)
    covariance = blob2numpy(covariance)

    print("--##    Mean shape: {}".format(mean.shape))
    print("--##    Covariance shape: {}".format(covariance.shape))

    epsilon = 1e-5
    variance = covariance ** 2 - epsilon

    node = ts.zoo.batch_norm(node_name, x=x, mean=mean, variance=variance, dim=1, epsilon=epsilon)

    return node,


def convert_batch_scale_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} ]=-".format(OPType.EnumString[layer.type], output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    param = layer.scale_param

    scale = param.scale_param
    bias = param.bias_param

    scale = blob2numpy(scale)
    bias = blob2numpy(bias)

    print("--##    Scale shape: {}".format(scale.shape))
    print("--##    Bias shape: {}".format(bias.shape))

    node = ts.zoo.batch_scale(node_name, x=x, scale=scale, bias=bias, dim=1)

    return node,


def convert_relu_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} ]=-".format(OPType.EnumString[layer.type], output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    node = ts.zoo.relu(node_name, x=x)

    return node,


def convert_pooling_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} ]=-".format(OPType.EnumString[layer.type], output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    param = layer.pooling_param

    pool = param.pool

    holiday_pool_type_to_ts_pool_type = {
        param.MAX: ts.zoo.Type.pooling_type.max,
        param.AVE: ts.zoo.Type.pooling_type.avg,
        param.STOCHASTIC: None,
    }

    holiday_pool_type_to_string = {
        param.MAX: "MAX",
        param.AVE: "AVE",
        param.STOCHASTIC: "STOCHASTIC",
    }

    print("--##    Type : {}".format(holiday_pool_type_to_string[pool]))

    type = holiday_pool_type_to_ts_pool_type[pool]
    if type is None:
        raise NotImplementedError("Pooling type(code): {}({})".format(holiday_pool_type_to_string[pool], pool))

    padding = [param.pad_height, param.pad_width]

    print("--##    Padding: {}".format(padding))

    kernel = [param.kernel_height, param.kernel_width]

    print("--##    Kernel: {}".format(kernel))

    stride = [param.stride_height, param.stride_width]

    print("--##    Stride: {}".format(stride))

    global_pooling = False
    if param.HasField("global_pooling"):
        global_pooling = param.global_pooling
        print("--##    Global pooling: {}".format(global_pooling))

    tf_padding = None
    if param.HasField("tf_padding"):
        tf_padding = param.tf_padding
        print("--##    TF padding: {}".format(tf_padding))

    valid = None
    if param.HasField("valid"):
        valid = param.valid
        print("--##    MX valid: {}".format(valid))

    assert tf_padding is None or tf_padding == "SAME" or tf_padding == "VALID"

    if global_pooling:
        raise NotImplementedError("Global pooling: {}".format(global_pooling))

    if tf_padding is not None:
        raise NotImplementedError("TF padding: {}".format(tf_padding))

    is_holiday = valid is None and tf_padding is None
    is_mxnet = valid is not None and tf_padding is None
    is_tf = valid is None and tf_padding is not None

    node = None

    if is_holiday:
        node = ts.zoo.pooling2d(node_name, x=x,
                                ksize=[1, 1, kernel[0], kernel[1]],
                                stride=[1, 1, stride[0], stride[1]],
                                type=type,
                                format=ts.zoo.Name.NCHW,
                                padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]])
    elif is_mxnet:
        node = ts.frontend.mxnet.pooling2d(node_name, x=x,
                                           ksize=[1, 1, kernel[0], kernel[1]],
                                           stride=[1, 1, stride[0], stride[1]],
                                           type=type,
                                           format=ts.zoo.Name.NCHW,
                                           padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
                                           valid=valid)

    if node is None:
        raise NotImplementedError(layer)

    return node,


def convert_inner_product_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} ]=-".format(OPType.EnumString[layer.type], output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    param = layer.inner_product_param

    num_output = None
    if param.HasField("num_output"):
        num_output = param.num_output

    axis = 1
    if param.HasField("axis"):
        axis = param.axis

    assert axis == 1

    transpose = False
    if param.HasField("transpose"):
        transpose = param.transpose
    print("--##    Transpose: {}".format(transpose))

    weights_blob = blob2numpy(param.Inner_param)
    print("--##    Weights shape: {}".format(weights_blob.shape))

    bias_blob = None
    if param.HasField("bias_param"):
        bias_blob = blob2numpy(param.bias_param)
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

    return node,


def convert_eltwise_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} ]=-".format(OPType.EnumString[layer.type], output_names))

    assert len(input_nodes) >= 2
    assert len(output_names) == 1

    x = input_nodes[0]
    node_name = output_names[0]

    param = layer.eltwise_param

    holiday_eltwise_op_to_ts_op = {
        param.PROD: ts.zoo.mul,
        param.SUM: ts.zoo.add,
        param.MAX: None,
    }

    holiday_eltwise_op_to_string = {
        param.PROD: "PROD",
        param.SUM: "SUM",
        param.MAX: "MAX",
    }

    operation = param.operation
    print("--##    Operation : {}".format(holiday_eltwise_op_to_string[operation]))

    coeff = param.coeff
    assert len(coeff) == 0 or len(coeff) == len(input_nodes)
    print("--##    Coeff shape: [{}, ]".format(len(coeff)))

    stable_prod_grad = True
    if param.HasField("stable_prod_grad"):
        stable_prod_grad = param.stable_prod_grad
    print("--##    Ignore param stable_prod_grad: {}".format(stable_prod_grad))

    ts_op = holiday_eltwise_op_to_ts_op[operation]

    if ts_op is None:
        NotImplementedError("Operation : {}".format(holiday_eltwise_op_to_string[operation]))

    is_sum = operation == param.SUM
    is_mul = operation == param.PROD
    is_max = operation == param.MAX

    node = None

    if is_mul:
        lhs = input_nodes[0]
        rhs = input_nodes[1:]
        for i in range(len(rhs)):
            lhs = ts.zoo.mul(name="_{}_{}".format(node_name, i), lhs=lhs, rhs=rhs[i])
        node = lhs
    elif is_sum:
        if len(coeff) > 0:
            for i in range(len(input_nodes)):
                node = input_nodes[i]
                input_nodes[i] = ts.zoo.mul(name="_coeff_{}_{}".format(node_name, i), lhs=node, rhs=coeff[i])
        lhs = input_nodes[0]
        rhs = input_nodes[1:]
        for i in range(len(rhs)):
            lhs = ts.zoo.add(name="_{}_{}".format(node_name, i), lhs=lhs, rhs=rhs[i])
        node = lhs
    elif is_max:
        NotImplementedError("Operation : {}".format(holiday_eltwise_op_to_string[operation]))

    if node is None:
        raise NotImplementedError(layer)

    node.name = node_name

    return node,


if __name__ == "__main__":
    # convert("test.ext.dat", "test.tsm", output_blobs=["fc_pose_umd"], has_header=True)
    convert("test.ext.dat", "test.tsm", export_all=True, has_header=True)
    #convert("/Users/seetadev/Documents/Files/models/VIPLFaceRecognizer5.0.RN30.light.ext.dat",
    #        "VIPLFaceRecognizer5.0.RN30.light.tsm", export_all=True, has_header=True)
