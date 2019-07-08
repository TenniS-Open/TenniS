#!python
# coding: UTF-8
"""
author: kier
"""

import tensorstack as ts

import tensorflow as tf
import numpy
from . import zipper


def tensor_to_numpy(x):
    # type: (tf.Tensor) -> numpy.ndarray
    return numpy.asarray(x.eval())


ts.tensor.register_dtype(tf.Tensor, to_numpy=tensor_to_numpy)


layer2converter = {
}


def register_layer_converter(layer, converter):
    layer2converter[layer] = converter


def convert(graph, inputs, outputs, output_file):
    if inputs is None:
        raise Exception("param #2 inputs must be set.")
    if outputs is None:
        raise Exception("param #3 outputs must be set.")
    if not isinstance(inputs, list) or not isinstance(inputs, tuple):
        inputs = (inputs, )
    if not isinstance(outputs, list) or not isinstance(outputs, tuple):
        outputs = (outputs, )

    set_no_log_converter = {
        convert_identity,
        convert_const,
    }

    # function format(node, inputs)
    map_converter = {
        # add layer converter here
        "Identity": convert_identity,
        "ConcatV2": convert_concat_v2,
        "Reshape": convert_reshape,
        "Sub": convert_sub,
        "Const": convert_const,
        "Placeholder": convert_placeholder,
        "RealDiv": convert_real_div,

        "StridedSlice": convert_not_implemented,
        "Pack": convert_not_implemented,
        "Pad": convert_pad,
        "AvgPool": convert_any_pool,
        "MaxPool": convert_any_pool,
        "Add": convert_add,
        "Mul": convert_mul,
        "BiasAdd": convert_bias_add,
        "Conv2D": convert_conv2d,
        "Relu": convert_relu,
        "Sum": convert_not_implemented,
        "Cast": convert_not_implemented,
        "Softmax": convert_not_implemented,
        "Shape": convert_not_implemented,

        "GatherV2": convert_not_implemented,
        "ResizeNearestNeighbor": convert_not_implemented,
        "Rsqrt": convert_rsqrt,
        "Maximum": convert_not_implemented,
        "Square": convert_not_implemented,
        "Range": convert_not_implemented,
        "Exp": convert_not_implemented,
        "Slice": convert_not_implemented,
        "TopKV2": convert_not_implemented,
        "Max": convert_not_implemented,
        "NonMaxSuppressionV3": convert_not_implemented,
        "ExpandDims": convert_not_implemented,
        "ArgMax": convert_not_implemented,

        # 2019-04-27
        "Squeeze": convert_squeeze,
        "Mean": convert_mean,
        "BatchToSpaceND": convert_batch_to_space_nd,
        "SpaceToBatchND": convert_space_to_batch_nd,

        # 2019-06-14
        # add in layer2converter
    }
    map_converter.update(layer2converter)

    # set_no_log_converter = set(map_converter.keys())

    map_tf_node_ts_node = {}

    output_ts_nodes = []

    def convert_node(tf_node):
        if tf_node in map_tf_node_ts_node:
            return map_tf_node_ts_node[tf_node]
        node_op = tf_node.op
        node_op_type = node_op.type
        if node_op_type not in map_converter:
            raise Exception("Not supported Layer: {}".format(node_op_type))
        converter = map_converter[node_op_type]

        input_ts_nodes = []
        for input in node_op.inputs:
            assert input != tf_node
            # TODO: checking loop
            ts_input_node = convert_node(input)
            # assert isinstance(ts_input_node, ts.Node)
            input_ts_nodes.append(ts_input_node)

        if converter not in set_no_log_converter:
            print("--# -=[ Converting {} layer: {} ]=-".format(tf_node.op.type, tf_node.op.name))

        # if converter == convert_not_implemented:
        #     return ts.menu.param("")

        ts_node = converter(tf_node, input_ts_nodes)
        if isinstance(ts_node, (tuple, list)):
            assert len(ts_node) == 1
            ts_node = ts_node[0]

        map_tf_node_ts_node[tf_node] = ts_node

        return ts_node

    for output in outputs:
        output_ts_nodes.append(convert_node(output))

    # print output_ts_nodes

    # checking inputs
    input_ts_nodes = []
    for input in inputs:
        if input not in map_tf_node_ts_node:
            raise Exception("Node {} not in graph".format(input.op.name))
        input_ts_nodes.append(map_tf_node_ts_node[input])

    # print input_ts_nodes

    """
    Reduce nchw operators
    """
    ready_zipped = {}
    ready_nchw = {}
    output_ts_nodes = [zipper.zipnode(output, ready_zipped=ready_zipped, ready_nchw=ready_nchw) for output in output_ts_nodes]
    #
    # zipper.plot_graph(output_ts_nodes)
    # output_ts_nodes = [zipper.warp_node(output) for output in output_ts_nodes]

    # saving
    inputs = input_ts_nodes
    outputs = output_ts_nodes

    module = ts.Module()

    # load module
    module.load(outputs)
    module.sort_inputs(inputs)

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

    return module


def convert_identity(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    return inputs[0]


def node_def_attr_dict(tf_node):
    # type: (tf.Tensor) -> dict
    # from collections import OrderedDict
    node_def = tf_node.op.node_def
    attr_dict = {}
    for attr in node_def.attr:
        # print "attr={}".format(attr)
        # print "type(attr)={}".format(type(attr))
        key = str(attr)
        value = tf_node.op.get_attr(key)
        if isinstance(value, bytes):
            value = value.decode()
        attr_dict[key] = value
    return attr_dict


def convert_concat_v2(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) > 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    N = attr_dict['N']

    if N != len(inputs) - 1:
        raise NotImplementedError("Concat N={} with {} inputs".format(N, len(inputs) - 1))

    axis = tf_node.op.inputs[N].eval()
    axis = numpy.asarray(axis, dtype=numpy.int32)

    # if axis < 0 or axis >= 4:
    #     raise NotImplementedError("Concat axis: {}".format(axis))

    return ts.zoo.concat(node_name, inputs[:-1], dim=axis)


def convert_pad(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]
    paddings = inputs[1]

    # paddings = ts.zoo.to_const(inputs[1], "inputs[1]")
    # paddings = ts.tensor.from_any(paddings, numpy.int32)

    known_count = 0
    if 'T' in attr_dict:
        known_count += 1
    if 'Tpaddings' in attr_dict:
        known_count += 1

    if len(attr_dict) != known_count:
        print("[WARNING] there has unchecked attr: {}".format(attr_dict))

    return ts.zoo.pad(node_name, x, paddings)


def conv2d_convert_weight(w):
    # type: (ts.Node) -> ts.Node
    if w.op == ts.Node.Const:
        value = w.params["value"]
        value = ts.tensor.from_any(value)
        value = value.transpose(3, 2, 0, 1)
        return ts.menu.data(w.name + "_nchw", value=value)
    else:
        return ts.zoo.transpose(w.name + "_nchw", x=w, permute=[3, 2, 0, 1])


def convert_conv2d(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]
    w = inputs[1]

    assert isinstance(x, ts.Node)
    static_padding = None
    if x.op == ts.zoo.Name.Layer.pad:
        x_padding = x.inputs[1]
        assert isinstance(x_padding, ts.Node)
        if x_padding.op == ts.Node.Const:
            static_padding = x_padding.get("value")
            static_padding = ts.tensor.from_any(static_padding, numpy.int32)
            x = x.inputs[0]
            assert static_padding.shape == (4, 2)
            print("--##    static_padding: {}".format(static_padding))

    w = conv2d_convert_weight(w)
    data_fromat = attr_dict['data_format']
    assert data_fromat == 'NHWC' or data_fromat == 'NCHW'

    dynamic_padding = attr_dict['padding']

    strides = [1, 1, 1, 1]
    if 'strides' in attr_dict:
        strides = attr_dict['strides']

    dilations = [1, 1, 1, 1]
    if 'dilations' in attr_dict:
        dilations = attr_dict['dilations']

    if data_fromat == 'NHWC':
        x = zipper.nhwc2nchw(x, name=x.name + "_nchw")
        if static_padding is not None:
            static_padding = numpy.take(static_padding, (0, 3, 1, 2), axis=0)
        strides = numpy.take(strides, (0, 3, 1, 2), axis=0)
        dilations = numpy.take(dilations, (0, 3, 1, 2), axis=0)

    node = ts.frontend.tf.conv2d(name=node_name + "_nchw", x=x, w=w,
                                 format=ts.zoo.Name.NCHW,
                                 padding=static_padding,
                                 padding_method=dynamic_padding,
                                 stride=strides,
                                 dilation=dilations)

    if data_fromat == 'NHWC':
        node = zipper.nchw2nhwc(x=node, name=node_name)
    else:
        node.name = node_name

    return node


def try_convert_fused_batch_norm(node):
    # type: (ts.Node) -> Union[ts.Node, None]
    add_1 = node

    assert isinstance(add_1, ts.Node)
    if add_1.op != ts.zoo.Name.Layer.add:
        return None
    mul_1 = add_1.inputs[0]
    sub = add_1.inputs[1]
    assert isinstance(mul_1, ts.Node)
    assert isinstance(sub, ts.Node)

    if sub.op != ts.zoo.Name.Layer.sub:
        return None
    beta = sub.inputs[0]
    mul_2 = sub.inputs[1]
    assert isinstance(beta, ts.Node)
    assert isinstance(mul_2, ts.Node)

    if mul_1.op != ts.zoo.Name.Layer.mul:
        return None
    x = mul_1.inputs[0]
    mul = mul_1.inputs[1]
    assert isinstance(x, ts.Node)
    assert isinstance(mul, ts.Node)

    if mul_2.op != ts.zoo.Name.Layer.mul:
        return None
    moving_mean = mul_2.inputs[0]
    mul_in_mul_2 = mul_2.inputs[1]
    assert isinstance(moving_mean, ts.Node)
    if mul_in_mul_2 != mul:
        return None

    if beta.op != ts.Node.Const:
        return None
    beta_value = ts.tensor.from_any(beta.get("value"))

    if moving_mean.op != ts.Node.Const:
        return None
    moving_mean_value = ts.tensor.from_any(moving_mean.get("value"))

    if mul.op != ts.zoo.Name.Layer.mul:
        return None
    rsqrt = mul.inputs[0]
    gamma = mul.inputs[1]
    assert isinstance(rsqrt, ts.Node)
    assert isinstance(gamma, ts.Node)

    if gamma.op != ts.Node.Const:
        return None
    gamma_value = ts.tensor.from_any(gamma.get("value"))

    if rsqrt.op != ts.zoo.Name.Layer.rsqrt:
        return None
    add = rsqrt.inputs[0]
    assert isinstance(add, ts.Node)

    if add.op != ts.zoo.Name.Layer.add:
        return None
    moving_variance = add.inputs[0]
    y = add.inputs[1]
    assert isinstance(moving_variance, ts.Node)
    assert isinstance(y, ts.Node)

    if y.op != ts.Node.Const:
        return None
    y_value = ts.tensor.from_any(y.get("value"))

    if moving_variance.op != ts.Node.Const:
        return None
    moving_variance_value = ts.tensor.from_any(moving_variance.get("value"))

    del add_1
    del sub
    del mul_1
    del mul_2
    del mul
    del mul_in_mul_2
    del rsqrt
    del add

    if len(moving_mean_value.shape) != 1 or \
        len(moving_variance_value.shape) != 1 or \
        len(beta_value.shape) != 1 or \
        len(gamma_value.shape) != 1 or \
        len(y_value.shape) != 0:
        return None

    batch_norm = ts.zoo.fused_batch_norm(name=node.name,
                                         x=x,
                                         mean=moving_mean,
                                         variance=moving_variance,
                                         scale=gamma,
                                         bias=beta,
                                         dim=3,
                                         epsilon=y_value)

    return batch_norm


def convert_add(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    lhs = inputs[0]
    rhs = inputs[1]

    node = ts.zoo.add(node_name, lhs, rhs)

    if len(tf_node.shape) == 4:
        batch_norm = try_convert_fused_batch_norm(node)
        if batch_norm is not None:
            print("--##    Is fused batch norm: {}".format(True))
            return batch_norm

    return node


def convert_rsqrt(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    return ts.zoo.rsqrt(node_name, x)


def convert_mul(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    lhs = inputs[0]
    rhs = inputs[1]

    node = ts.zoo.mul(node_name, lhs, rhs)

    return node


def convert_relu(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    return ts.zoo.relu(node_name, x)


def convert_any_pool(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    assert isinstance(x, ts.Node)
    static_padding = None
    if x.op == ts.zoo.Name.Layer.pad:
        x_padding = x.inputs[1]
        assert isinstance(x_padding, ts.Node)
        if x_padding.op == ts.Node.Const:
            static_padding = x_padding.get("value")
            static_padding = ts.tensor.from_any(static_padding, numpy.int32)
            x = x.inputs[0]
            assert static_padding.shape == (4, 2)
            print("--##    static_padding: {}".format(static_padding))

    data_fromat = attr_dict['data_format']
    assert data_fromat == 'NHWC' or data_fromat == 'NCHW'

    dynamic_padding = attr_dict['padding']

    strides = [1, 1, 1, 1]
    if 'strides' in attr_dict:
        strides = attr_dict['strides']

    ksize = attr_dict['ksize']

    if data_fromat == 'NHWC':
        x = zipper.nhwc2nchw(x, name=x.name + "_nchw")
        if static_padding is not None:
            static_padding = numpy.take(static_padding, (0, 3, 1, 2), axis=0)
        strides = numpy.take(strides, (0, 3, 1, 2), axis=0)
        ksize = numpy.take(ksize, (0, 3, 1, 2), axis=0)

    pooling_type_map = {
        "AvgPool": ts.zoo.Type.pooling_type.avg,
        "MaxPool": ts.zoo.Type.pooling_type.max,
    }

    if tf_node.op.type not in pooling_type_map:
        raise NotImplementedError("node.op.type={}".format(tf_node.op.type))

    pooling_type = pooling_type_map[tf_node.op.type]

    node = ts.frontend.tf.pooling2d(name=node_name + "_nchw", x=x,
                                    ksize=ksize,
                                    stride=strides,
                                    type=pooling_type,
                                    format=ts.zoo.Name.NCHW,
                                    padding=static_padding,
                                    padding_method=dynamic_padding)

    if data_fromat == 'NHWC':
        node = zipper.nchw2nhwc(x=node, name=node_name)
    else:
        node.name = node_name

    return node


def convert_space_to_batch_nd(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 3
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    block_shape = inputs[1]
    padding = inputs[2]

    block_shape = ts.zoo.to_const(block_shape, "block_shape")
    padding = ts.zoo.to_const(padding, "padding")
    block_shape = ts.tensor.from_any(block_shape, dtype=numpy.int32)
    padding = ts.tensor.from_any(padding, dtype=numpy.int32)

    x = zipper.nhwc2nchw(x=x, name=x.name + "_nchw")

    node = ts.frontend.tf.space_to_batch4d(name=node_name + "_nchw", x=x, block_shape=block_shape, padding=padding)

    node = zipper.nchw2nhwc(x=node, name=node_name)

    return node


def convert_batch_to_space_nd(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 3
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    block_shape = inputs[1]
    crop = inputs[2]

    block_shape = ts.zoo.to_const(block_shape, "block_shape")
    crop = ts.zoo.to_const(crop, "crop")
    block_shape = ts.tensor.from_any(block_shape, dtype=numpy.int32)
    crop = ts.tensor.from_any(crop, dtype=numpy.int32)

    x = zipper.nhwc2nchw(x=x, name=x.name + "_nchw")

    node = ts.frontend.tf.batch_to_space4d(name=node_name + "_nchw", x=x, block_shape=block_shape, crop=crop)

    node = zipper.nchw2nhwc(x=node, name=node_name)

    return node


def convert_bias_add(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]
    bias = inputs[1]

    data_fromat = attr_dict['data_format']
    assert data_fromat == 'NHWC' or data_fromat == 'NCHW'

    if data_fromat == 'NHWC':
        x = zipper.nhwc2nchw(x, name=x.name + "_nchw")

    node = ts.zoo.add_bias(name=node_name + "_nchw", x=x, b=bias, dim=1)

    if data_fromat == 'NHWC':
        node = zipper.nchw2nhwc(x=node, name=node_name)
    else:
        node.name = node_name

    return node


def convert_not_implemented(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    attr_dict = node_def_attr_dict(tf_node)

    raise NotImplementedError("{} with attr: {}".format(tf_node.op.type, attr_dict))


def convert_strided_slice(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    raise NotImplementedError(tf_node.op.type)


def convert_reshape(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2

    shape = tf_node.op.inputs[1].eval()
    shape = numpy.asarray(shape, dtype=numpy.int32)

    return ts.zoo.reshape(tf_node.op.name, inputs[0], shape=shape)


def convert_sub(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2

    node_name = tf_node.op.name

    return ts.zoo.sub(node_name, lhs=inputs[0], rhs=inputs[1])


def convert_real_div(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2

    node_name = tf_node.op.name

    return ts.zoo.div(node_name, lhs=inputs[0], rhs=inputs[1])


def convert_placeholder(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 0

    return ts.menu.param(name=tf_node.op.name)


def convert_const(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 0

    return ts.menu.data(name=tf_node.op.name, value=tf_node)


def convert_squeeze(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    squeeze_dims = []
    if "squeeze_dims" in attr_dict:
        squeeze_dims = attr_dict["squeeze_dims"]

    return ts.zoo.squeeze(node_name, x=inputs[0], axes=squeeze_dims)


def convert_mean(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    keep_dims = attr_dict["keep_dims"]
    reduction_indices = ts.zoo.to_const(inputs[1], "inputs[1]")
    reduction_indices = ts.tensor.from_any(reduction_indices, dtype=numpy.int32)

    if keep_dims is False:
        raise NotImplementedError("Mean keep_dims: %s" % str(keep_dims))

    if len(reduction_indices) != 2 or reduction_indices[0] != 1 or reduction_indices[1] != 2:
        raise NotImplementedError("Mean reduction_indices: %s" % str(reduction_indices))

    # now mean is global avg pooling with NHWC format
    data_fromat = 'NHWC'

    if data_fromat == 'NHWC':
        x = zipper.nhwc2nchw(x, name=x.name + "_nchw")

    node = ts.zoo.global_pooling2d(name=node_name + "_nchw",
                                   x=x, type=ts.zoo.Type.pooling_type.avg,
                                   format=ts.zoo.Name.NCHW)

    if data_fromat == 'NHWC':
        node = zipper.nchw2nhwc(x=node, name=node_name)
    else:
        node.name = node_name

    return node


def convert_fused_batch_norm(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 5
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]
    gamma = inputs[1]
    beta = inputs[2]
    moving_mean = inputs[3]
    moving_variance = inputs[4]

    data_fromat = attr_dict['data_format']
    assert data_fromat == 'NHWC' or data_fromat == 'NCHW'

    epsilon = attr_dict['epsilon']

    if data_fromat == 'NHWC':
        x = zipper.nhwc2nchw(x, name=x.name + "_nchw")


    node = ts.zoo.fused_batch_norm(name=node_name, x=x,
                                   mean=moving_mean, variance=moving_variance, scale=gamma, bias=beta,
                                   dim=1, epsilon=epsilon)

    if data_fromat == 'NHWC':
        node = zipper.nchw2nhwc(x=node, name=node_name)
    else:
        node.name = node_name

    return node


register_layer_converter("FusedBatchNorm", convert_fused_batch_norm)


def convert_relu6(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    return ts.zoo.relu_max(node_name, x, max=6)


register_layer_converter("Relu6", convert_relu6)


def depthwise_conv2d_convert_weight(w):
    # type: (ts.Node) -> ts.Node
    if w.op == ts.Node.Const:
        value = w.params["value"]
        value = ts.tensor.from_any(value)
        value = value.transpose(3, 2, 0, 1)
        return ts.menu.data(w.name + "_nchw", value=value)
    else:
        return ts.zoo.transpose(w.name + "_nchw", x=w, permute=[3, 2, 0, 1])


def convert_depthwise_conv2d_native(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 2
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]
    w = inputs[1]

    assert isinstance(x, ts.Node)
    static_padding = None
    if x.op == ts.zoo.Name.Layer.pad:
        x_padding = x.inputs[1]
        assert isinstance(x_padding, ts.Node)
        if x_padding.op == ts.Node.Const:
            static_padding = x_padding.get("value")
            static_padding = ts.tensor.from_any(static_padding, numpy.int32)
            x = x.inputs[0]
            assert static_padding.shape == (4, 2)
            print("--##    static_padding: {}".format(static_padding))

    w = depthwise_conv2d_convert_weight(w)
    data_fromat = attr_dict['data_format']
    assert data_fromat == 'NHWC' or data_fromat == 'NCHW'

    dynamic_padding = attr_dict['padding']

    strides = [1, 1, 1, 1]
    if 'strides' in attr_dict:
        strides = attr_dict['strides']

    dilations = [1, 1, 1, 1]
    if 'dilations' in attr_dict:
        dilations = attr_dict['dilations']

    if data_fromat == 'NHWC':
        x = zipper.nhwc2nchw(x, name=x.name + "_nchw")
        if static_padding is not None:
            static_padding = numpy.take(static_padding, (0, 3, 1, 2), axis=0)
        strides = numpy.take(strides, (0, 3, 1, 2), axis=0)
        dilations = numpy.take(dilations, (0, 3, 1, 2), axis=0)

    node = ts.frontend.tf.depthwise_conv2d(name=node_name + "_nchw", x=x, w=w,
                                           format=ts.zoo.Name.NCHW,
                                           padding=static_padding,
                                           padding_method=dynamic_padding,
                                           stride=strides,
                                           dilation=dilations)

    if data_fromat == 'NHWC':
        node = zipper.nchw2nhwc(x=node, name=node_name)
    else:
        node.name = node_name

    return node


register_layer_converter("DepthwiseConv2dNative", convert_depthwise_conv2d_native)


def convert_tensor_gather_v3(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    raise NotImplementedError("type={}".format(tf_node.op.type))


register_layer_converter("TensorArrayGatherV3", convert_tensor_gather_v3)


def convert_tensor_array_size_v3(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    raise NotImplementedError("type={}".format(tf_node.op.type))


register_layer_converter("TensorArraySizeV3", convert_tensor_array_size_v3)


def convert_exit(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    raise NotImplementedError("type={}".format(tf_node.op.type))


register_layer_converter("Exit", convert_exit)


def convert_switch(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    raise NotImplementedError("type={}".format(tf_node.op.type))


register_layer_converter("Switch", convert_switch)


def convert_merge(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    raise NotImplementedError("type={}".format(tf_node.op.type))


register_layer_converter("Merge", convert_merge)


def convert_enter(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    raise NotImplementedError("type={}".format(tf_node.op.type))

    x = inputs[0]

    node = ts.menu.op(name=node_name, op_name="_enter", inputs=inputs)

    return node


register_layer_converter("Enter", convert_enter)


def convert_tensor_array_v3(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    attr_dict = node_def_attr_dict(tf_node)
    print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    dtype = attr_dict["dtype"]

    numpy_dtype = dtype.as_numpy_dtype

    x = inputs[0]

    return ts.zoo.cast(name=node_name, x=x, dtype=numpy_dtype)


register_layer_converter("TensorArrayV3", convert_tensor_array_v3)


def convert_shape(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 1
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]

    return ts.zoo.shape(name=node_name, x=x)


register_layer_converter("Shape", convert_shape)


def convert_stirded_slice(tf_node, inputs):
    # type: (tf.Tensor, List[ts.Node]) -> ts.Node

    assert len(inputs) == 4
    # attr_dict = node_def_attr_dict(tf_node)
    # print("--##    attr: {}".format(attr_dict))
    node_name = tf_node.op.name

    x = inputs[0]
    begin = inputs[1]
    end = inputs[2]
    stride = inputs[3]

    return ts.frontend.tf.strided_slice(name=node_name, x=x, begin=begin, end=end, stride=stride)


register_layer_converter("StridedSlice", convert_stirded_slice)




