#!python
# coding: UTF-8
"""
author: kier
"""

import mxnet as mx
import json

from . import parser

import tensorstack as ts
import numpy


layer2converter = {
}


def register_layer_converter(layer, converter):
    layer2converter[layer] = converter


def convert(model_prefix, epoch, output_file,
            input_shapes=None, input_nodes=None,
            output_node_names=None):
    """
    :param model_prefix: string of model prefix
    :param epoch: int
    :param output_file: path to output file
    :param input_shapes: dict of string to shape, like {"data", [1, 3, 248, 248]}.
    :param input_nodes: dict of string to ts.Node, like {"data", ts.Node()}.
    :param output_node_names: list of string to get node.
    :return: ts.Module
    """
    symbol_json = '%s-symbol.json' % (model_prefix, )
    symbol = None
    with open(symbol_json) as f:
        symbol = json.load(f)

    _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)

    graph = parser.Graph(symbol, arg_params, aux_params)

    if input_nodes is None:
        input_nodes = {}
    if input_shapes is None:
        input_shapes = {}
    for name, input_shape in input_shapes.items():
        if isinstance(input_shape, ts.Node):
            input_nodes[name] = input_shape
        else:
            input_nodes[name] = ts.menu.param(name, input_shape)

    # make sure all input are nodes
    for name in input_nodes.keys():
        input_node = input_nodes[name]
        if isinstance(input_node, ts.Node):
            continue
        input_nodes[name] = ts.menu.param(name, input_node)

    output_node_name2index = {}
    output_node_name_set = set()
    if output_node_names is None:
        # if no output_node_names, then make output checking always failed
        output_node_name2index["^_^"] = -1
    else:
        output_node_name_set = set(output_node_names)

    def convert_null(node, inputs):
        assert len(inputs) == 0
        name = node["name"]
        if name in input_nodes:
            input_shape = []
            if name in input_shapes:
                input_shape = input_shapes[name]
            print("--# -=[ Placeholder: {}, {} ]=-".format(name, input_shape))
            return input_nodes[name]
        param = graph.param(name)
        if param is not None:
            param = param.asnumpy()
            print("--# -=[ Load data: {}, {} ]=-".format(name, param.shape))
            return ts.menu.data(name, param)
        raise Exception("Can not load param: {}".format(name))

    def convert_not_implemented(node, inputs):
        # type: (dict, List[ts.Node]) -> ts.Node
        print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

        raise NotImplementedError(node["op"])

    # function format(node, inputs)
    converter_map = {
        "null": convert_null,
        # add layer converter here
        "BatchNorm": convert_batch_norm,
        "Convolution": convert_convolution,
        "Activation": convert_activation,
        "Pooling": convert_pooling,
        "elemwise_add": convert_elemwise_add,
        "Flatten": convert_flatten,
        "FullyConnected": convert_fully_connected,
    }
    converter_map.update(layer2converter)

    ts_nodes = [None] * len(graph.nodes)

    def convert_at(i):
        """
        :param i: node index
        :return: true if early return
        """
        if ts_nodes[i] is not None:
            return
        node = graph.nodes[i]
        # name = node["name"]
        op = node["op"]
        if op not in converter_map:
            raise Exception("Not supported Layer: {}".format(op))
        converter = converter_map[op]
        inputs = [ input[0] for input in node["inputs"] ]
        for input in inputs:
            assert input != i
            # TODO: checking loop
            convert_at(input)
        inputs = [ ts_nodes[input] for input in inputs ]
        local_nodes = converter(node, inputs)
        if isinstance(local_nodes, (tuple, list)):
            assert len(local_nodes) == 1
            local_nodes = local_nodes[0]
        ts_nodes[i] = local_nodes
        # check if output
        if node['name'] in output_node_name_set:
            output_node_name2index[node['name']] = i
        return len(output_node_name2index) == len(output_node_name_set)

    for i in range(len(graph.nodes)):
        if convert_at(i):
            break

    heads = []
    if output_node_names is not None:
        heads = [output_node_name2index[name] for name in output_node_names]
    else:
        heads = graph.heads

    outputs = [ts_nodes[head] for head in heads]

    module = ts.Module()

    # load module
    module.load(outputs)

    # sort inputs
    assert len(module.inputs) == 1

    with open(output_file, "wb") as fo:
        ts.Module.Save(stream=fo, module=module)

    print("============ Summary ============")
    # print("Input file: {}".format(input_file))
    print("Input model_prefix: {}".format(model_prefix))
    print("Input epoch: {}".format(epoch))
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


def convert_batch_norm(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 5

    x = inputs[0]
    gamma = inputs[1]
    beta = inputs[2]
    moving_mean = inputs[3]
    moving_var = inputs[4]

    attrs = node["attrs"]

    eps = 1e-5
    if "eps" in attrs:
        eps = float(attrs["eps"])
        print("--##    eps: {}".format(eps))

    fix_gamma = True
    if "fix_gamma" in attrs:
        fix_gamma = attrs["fix_gamma"] == "True"
        print("--##    fix_gamma: {}".format(fix_gamma))

    if fix_gamma:
        fixed_gamma = ts.zoo.to_const(gamma, "gamma")
        gamma = numpy.ones_like(fixed_gamma, dtype=fixed_gamma.dtype)

    node = ts.zoo.fused_batch_norm(node["name"], x=x, mean=moving_mean, variance=moving_var,
                                   scale=gamma, bias=beta, dim=1, epsilon=eps)

    return node


def parse_tuple(string):
    # parse '(1, 2)'
    length = len(string)
    string = string[1: -1]
    str_list = string.split(',')
    return [int(i) for i in str_list]


def convert_convolution(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 2 or len(inputs) == 3

    conv2d_name = "_conv2d_" + node["name"]
    bias_name = "_bias_" + node["name"]
    node_name = node["name"]

    attrs = node["attrs"]

    no_bias = False
    if "no_bias" in attrs:
        no_bias = attrs["no_bias"] == "True"
        print("--##    no_bias: {}".format(no_bias))

    x = inputs[0]
    weight = inputs[1]
    bias = None

    weights_blob = ts.zoo.to_const(weight, "inputs[1]")
    if not no_bias:
        bias = inputs[2]

    num_filter = int(attrs["num_filter"])

    num_group = 1
    if "num_group" in attrs:
        num_group = int(attrs['num_group'])

    padding = (0, 0)
    if "pad" in attrs:
        padding = parse_tuple(attrs['pad'])

    kernel = parse_tuple(attrs['kernel'])
    stride = parse_tuple(attrs['stride'])

    dilation = (1, 1)
    if "dilation" in attrs:
        dilation = parse_tuple(attrs['dilation'])

    print("--##    dilation: {}".format(dilation))
    print("--##    pad: {}".format(padding))
    print("--##    stride: {}".format(stride))
    print("--##    kernel: {}".format(kernel))

    activation = None
    if "activation" in attrs:
        activation = attrs['activation']

    if activation is not None:
        raise NotImplementedError("activation = {}".format(activation))

    if num_group != 1 and weights_blob.shape[1] != 1:
        raise NotImplementedError("group = {} with weights.shape[1] = {}".format(num_group, weights_blob.shape[1]))

    assert weights_blob.shape[0] == num_filter

    is_conv2d = num_group == 1
    is_depthwise_conv2d = weights_blob.shape[1] == 1

    if is_depthwise_conv2d:
        assert weights_blob.shape[1] * num_group == num_filter

    node = None

    if is_conv2d:
        node = ts.zoo.conv2d(conv2d_name, x=x, w=weight, format=ts.zoo.Name.NCHW,
                             padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
                             padding_value=0,
                             stride=[1, 1, stride[0], stride[1]],
                             dilation=[1, 1, dilation[0], dilation[1]])
    elif is_depthwise_conv2d:
        weights_shape = weights_blob.shape
        depthwise_weights_shape = (weights_shape[1], weights_shape[0], weights_shape[2], weights_shape[3])
        weights_blob = numpy.reshape(weights_blob, depthwise_weights_shape)
        node = ts.zoo.depthwise_conv2d(conv2d_name, x=x, w=weights_blob, format=ts.zoo.Name.NCHW,
                                       padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
                                       padding_value=0,
                                       stride=[1, 1, stride[0], stride[1]],
                                       dilation=[1, 1, dilation[0], dilation[1]])

    if node is None:
        raise NotImplementedError(node)

    if bias is not None:
        node = ts.zoo.add_bias(bias_name, x=node, b=bias, format=ts.zoo.Name.NCHW)

    node.name = node_name

    return node


def convert_activation(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 1

    node_name = node["name"]

    attrs = node["attrs"]

    x = inputs[0]

    act_type = attrs['act_type']
    print("--##    act_type: {}".format(act_type))

    if act_type == 'relu':
        return ts.zoo.relu(node_name, x=x)
    elif act_type == 'sigmoid':
        return ts.zoo.sigmoid(node_name, x=x)
    else:
        raise NotImplementedError("act_type = {}".format(act_type))


def convert_pooling(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 1

    node_name = node["name"]

    attrs = node["attrs"]

    x = inputs[0]

    pool_type = attrs['pool_type']

    mx_pool_type_to_ts_pool_type = {
        "max": ts.zoo.Type.pooling_type.max,
        "avg": ts.zoo.Type.pooling_type.avg,
    }

    ts_pool_type = None
    if pool_type in mx_pool_type_to_ts_pool_type:
        ts_pool_type = mx_pool_type_to_ts_pool_type[pool_type]

    if ts_pool_type is None:
        raise NotImplementedError("pool_type = {}".format(pool_type))

    global_pool = False
    if "global_pool" in attrs:
        global_pool = attrs['global_pool'] == 'True'

    if global_pool:
        node = ts.zoo.global_pooling2d(node_name, x=x, type=ts_pool_type, format=ts.zoo.Name.NCHW)
        return node

    padding = (0, 0)
    if "pad" in attrs:
        padding = parse_tuple(attrs['pad'])

    kernel = parse_tuple(attrs['kernel'])

    stride = (1, 1)
    if "stride" in attrs:
        stride = parse_tuple(attrs['stride'])

    print("--##    pad: {}".format(padding))
    print("--##    stride: {}".format(stride))
    print("--##    kernel: {}".format(kernel))

    valid = True
    if "valid" in attrs:
        valid = attrs['valid'] == 'True'

    node = ts.frontend.mxnet.pooling2d(name=node_name, x=x,
                                       ksize=[1, 1, kernel[0], kernel[1]],
                                       stride=[1, 1, stride[0], stride[1]],
                                       type=ts_pool_type, format=ts.zoo.Name.NCHW,
                                       padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
                                       valid=valid)

    return node


def convert_elemwise_add(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) > 0

    node_name = node['name']

    lhs = inputs[0]
    rhs = inputs[1:]
    for i in range(len(rhs)):
        lhs = ts.zoo.add(name="_{}_{}".format(node_name, i), lhs=lhs, rhs=rhs[i])
    node = lhs

    node.name = node_name

    return node


def convert_flatten(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 1

    node_name = node['name']

    x = inputs[0]

    return ts.zoo.flatten(name=node_name, x=x, dim=1)


def convert_fully_connected(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 2 or len(inputs) == 3

    node_name = node['name']

    attrs = node['attrs']

    no_bias = False
    if "no_bias" in attrs:
        no_bias = attrs["no_bias"] == "True"
        print("--##    no_bias: {}".format(no_bias))

    x = inputs[0]
    weight = inputs[1]
    bias = None

    if not no_bias:
        assert len(inputs) == 3
        bias = inputs[2]

    weight_value = None
    try:
        weight_value = ts.zoo.to_const(weight, "weight")
    except Exception as e:
        pass

    x = ts.zoo.flatten(name=node_name + "_flatten", x=x)

    node = None
    if weight_value is None:
        weight_t = ts.zoo.transpose(name="_t_" + node_name, x=weight)
        node = ts.zoo.inner_prod(name="_ip_" + node_name, lhs=x, rhs=weight_t)
    else:
        node = ts.zoo.inner_prod(name="_ip_" + node_name, lhs=x, rhs=weight_value.T)

    if bias is not None:
        node = ts.zoo.add_bias(name="_bias_" + node_name, x=node, b=bias, dim=1)

    node.name = node_name

    return node


def convert_copy(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 1

    node_name = node['name']

    attrs = node['attrs']

    node = ts.zoo.copy(name=node_name, x=inputs[0])

    return node


register_layer_converter("_copy", convert_copy)


def convert_minus_scalar(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 1

    node_name = node['name']

    attrs = node['attrs']

    scalar = float(attrs['scalar'])

    node = ts.zoo.sub(name=node_name, lhs=inputs[0], rhs=scalar, dtype=numpy.float32)

    return node


register_layer_converter("_minus_scalar", convert_minus_scalar)


def convert_mul_scalar(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 1

    node_name = node['name']

    attrs = node['attrs']

    scalar = float(attrs['scalar'])

    node = ts.zoo.mul(name=node_name, lhs=inputs[0], rhs=scalar, dtype=numpy.float32)

    return node


register_layer_converter("_mul_scalar", convert_mul_scalar)


def convert_contrib_resize2d(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 1

    node_name = node['name']

    attrs = node['attrs']

    height = int(attrs['height'])
    width = int(attrs['width'])

    node = ts.zoo.resize2d(name=node_name, x=inputs[0], size=[-1, -1, height, width],
                           type=ts.zoo.Type.resize2d_type.linear)

    return node


register_layer_converter("_contrib_BilinearResize2D", convert_contrib_resize2d)


def convert_LeakyReLU(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 2

    node_name = node['name']

    attrs = node['attrs']

    act_type = attrs['act_type']
    if act_type != "prelu":
        raise NotImplementedError("act_type = {}".format(act_type))

    a = ts.zoo.to_const(inputs[1])

    node = ts.zoo.prelu(name=node_name, x=inputs[0], dim=1, slope=inputs[1])

    return node


register_layer_converter("LeakyReLU", convert_LeakyReLU)


def convert_Concat(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    # assert len(inputs) == 2

    node_name = node['name']

    attrs = node['attrs']

    dim = int(attrs['dim'])

    num_args = len(inputs)
    if 'num_args' in attrs:
        num_args = int(attrs['num_args'])

    assert len(inputs) == num_args

    node = ts.zoo.concat(name=node_name, inputs=inputs, dim=dim)

    return node


register_layer_converter("Concat", convert_Concat)


register_layer_converter("Dropout", convert_copy)


def convert_l2_norm(node, inputs):
    # type: (dict, List[ts.Node]) -> ts.Node
    print("--# -=[ Converting {} layer: {} ]=-".format(node["op"], node["name"]))

    assert len(inputs) == 1

    node_name = node['name']

    attrs = node['attrs']

    mode = attrs['mode']

    if mode != 'instance':
        raise NotImplementedError("mode = {}".format(mode))
    eps = 1.00000001e-10
    if 'eps' in attrs:
        eps = float(attrs['eps'])

    x = inputs[0]
    x_shape = ts.zoo.shape(name=node_name + "_shape", x=x)
    x_flatten = ts.zoo.flatten(name=node_name + "_flatten", x=x)
    x_l2 = ts.zoo.l2_norm(name=node_name + "_L2", x=x_flatten, dim=-1, epsilon=eps)
    node = ts.zoo.reshape_v2(name=node_name, x=x_l2, shape=x_shape)

    return node


register_layer_converter("L2Normalization", convert_l2_norm)


