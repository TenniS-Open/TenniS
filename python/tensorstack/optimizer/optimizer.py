from .. import Node
from .. import zoo
from .. import tensor
from .. import menu

import sys
import numpy
import copy


def __fused_batch_norm(w, b, mean, variance, epsilon=1e-5, w_shape=None):
    if w_shape is None:
        w_shape = (variance.shape[0], 1, 1, 1)
    d = variance + epsilon
    std = numpy.sqrt(variance + epsilon)
    w = w / numpy.reshape(std, newshape=w_shape)
    b = (b - mean) / std
    return w, b


def __fused_batch_scale(w, b, scale, bias, w_shape=None):
    if w_shape is None:
        w_shape = (scale.shape[0], 1, 1, 1)
    w = w * numpy.reshape(scale, newshape=w_shape)
    b = b * scale + bias
    return w, b


def __fused_conv_bn(name, conv2d, add_bias=None, fused_batch_norm=None, batch_norm=None, batch_scale=None):
    # type: (str, Node, Node, Node, Node, Node) -> Union[Node, None]
    if fused_batch_norm is None and batch_norm is None and batch_scale is None:
        return None

    x = conv2d.inputs[0]

    channels = None
    w = None
    w_shape = None
    w_input_index = 1
    V2 = False
    if conv2d.op == zoo.Name.Layer.conv2d:
        w = conv2d.inputs[1]
        if w.op != Node.Const:
            return None
        w = tensor.from_any(w.get("value"))
        channels = w.shape[0]
        w_shape = (w.shape[0], 1, 1, 1)
    elif conv2d.op == zoo.Name.Layer.depthwise_conv2d:
        w = conv2d.inputs[1]
        if w.op != Node.Const:
            return None
        w = tensor.from_any(w.get("value"))
        channels = w.shape[0] * w.shape[1]
        w_shape = (w.shape[0], w.shape[1], 1, 1)
    elif conv2d.op == zoo.Name.Layer.conv2d_v2:
        w = conv2d.inputs[2]
        if w.op != Node.Const:
            return None
        w = tensor.from_any(w.get("value"))
        channels = w.shape[0]
        w_shape = (w.shape[0], 1, 1, 1)
        w_input_index = 2
        V2 = True
    elif conv2d.op == zoo.Name.Layer.depthwise_conv2d_v2:
        w = conv2d.inputs[2]
        if w.op != Node.Const:
            return None
        w = tensor.from_any(w.get("value"))
        channels = w.shape[0] * w.shape[1]
        w_shape = (w.shape[0], w.shape[1], 1, 1)
        w_input_index = 2
        V2 = True
    else:
        raise NotImplementedError("Not supported op: {}".format(conv2d.op))

    b = None
    if add_bias is None:
        b = numpy.zeros(shape=(channels,), dtype=w.dtype)
        add_bias = zoo.add_bias(name=name + "_bias", x=conv2d, b=b, dim=1)

    else:
        b = add_bias.inputs[1]
        if b.op != Node.Const:
            return None
        b = tensor.from_any(b.get("value"))

    if fused_batch_norm is not None:
        mean = fused_batch_norm.inputs[1]
        variance = fused_batch_norm.inputs[2]
        scale = fused_batch_norm.inputs[3]
        bias = fused_batch_norm.inputs[4]
        if mean.op != Node.Const or variance.op != Node.Const or scale.op != Node.Const or bias.op != Node.Const:
            return None
        mean = tensor.from_any(mean.get("value"))
        variance = tensor.from_any(variance.get("value"))
        scale = tensor.from_any(scale.get("value"))
        bias = tensor.from_any(bias.get("value"))
        epsilon = float(fused_batch_norm.get("epsilon"))
        w, b = __fused_batch_norm(w=w, b=b, mean=mean, variance=variance, epsilon=epsilon,
                                  w_shape=w_shape)
        w, b = __fused_batch_scale(w=w, b=b, scale=scale, bias=bias,
                                   w_shape=w_shape)

    if batch_norm is not None:
        mean = batch_norm.inputs[1]
        variance = batch_norm.inputs[2]
        if mean.op != Node.Const or variance.op != Node.Const:
            return None
        mean = tensor.from_any(mean.get("value"))
        variance = tensor.from_any(variance.get("value"))
        epsilon = float(batch_norm.get("epsilon"))
        w, b = __fused_batch_norm(w=w, b=b, mean=mean, variance=variance, epsilon=epsilon,
                                  w_shape=w_shape)

    if batch_scale is not None:
        scale = batch_scale.inputs[1]
        bias = batch_scale.inputs[2]
        if scale.op != Node.Const or bias.op != Node.Const:
            return None
        scale = tensor.from_any(scale.get("value"))
        bias = tensor.from_any(bias.get("value"))
        w, b = __fused_batch_scale(w=w, b=b, scale=scale, bias=bias,
                                   w_shape=w_shape)

    new_conv2d = copy.copy(conv2d)
    new_bias = copy.copy(add_bias)
    new_conv2d_inputs = [i for i in new_conv2d.inputs]
    new_bias_inputs = [i for i in new_bias.inputs]

    import re
    if V2 and re.match(".*_conv2d_padding", new_conv2d_inputs[1].op):
        new_dynamic_padding = copy.copy(new_conv2d_inputs[1])
        new_dynamic_padding_inputs = [i for i in new_dynamic_padding.inputs]
        new_conv2d_weights = menu.data(name=name + "_weights", value=w)
        new_dynamic_padding_inputs[1] = new_conv2d_weights
        new_conv2d_inputs[0] = x
        new_conv2d_inputs[1] = new_dynamic_padding
        new_conv2d_inputs[2] = new_conv2d_weights
        new_bias_inputs[0] = new_conv2d
        new_bias_inputs[1] = menu.data(name=name + "_bias", value=b)
        Node.Link(new_dynamic_padding, new_dynamic_padding_inputs)
        Node.Link(new_conv2d, new_conv2d_inputs)
        Node.Link(new_bias, new_bias_inputs)
    else:
        new_conv2d_inputs[0] = x
        new_conv2d_inputs[w_input_index] = menu.data(name=name + "_weights", value=w)
        new_bias_inputs[0] = new_conv2d
        new_bias_inputs[1] = menu.data(name=name + "_bias", value=b)
        Node.Link(new_conv2d, new_conv2d_inputs)
        Node.Link(new_bias, new_bias_inputs)

    new_conv2d.name = conv2d.name + "_with_bn"
    new_bias.name = name

    return new_bias


def fused_conv_bn_convert(node, ready=None):
    # type: (Node, dict) -> Node
    if ready is None:
        ready = {}

    if node in ready:
        return ready[node]

    fused_conv_bn_param = [None] * 5
    fused_conv_bn_param_i = 5
    fused_conv_bn_param_dict = {
        zoo.Name.Layer.batch_scale: 4,
        zoo.Name.Layer.batch_norm: 3,
        zoo.Name.Layer.fused_batch_norm: 2,
        zoo.Name.Layer.add_bias: 1,
        zoo.Name.Layer.conv2d: 0,
        zoo.Name.Layer.conv2d_v2: 0,
        zoo.Name.Layer.depthwise_conv2d: 0,
        zoo.Name.Layer.depthwise_conv2d_v2: 0,
    }

    anchor = node
    while True:
        if anchor.op not in fused_conv_bn_param_dict:
            break
        index = fused_conv_bn_param_dict[anchor.op]
        if index >= fused_conv_bn_param_i:
            break
        fused_conv_bn_param_i = index
        fused_conv_bn_param[index] = anchor
        anchor = anchor.inputs[0]

    if fused_conv_bn_param[0] is not None:
        fused_node = __fused_conv_bn(node.name, *fused_conv_bn_param)
        if fused_node is not None:
            ready[node] = fused_node
            node = fused_node

    fused_inputs = []
    for input in node.inputs:
        fused_inputs.append(fused_conv_bn_convert(input, ready=ready))

    Node.Link(node, fused_inputs)

    ready[node] = node

    return node


def optimize(nodes, **kwargs):
    # type: (Union[List[Node], Node], ...) -> List[Node]
    """
    fused_conv_bn=True
    :param nodes:
    :param kwargs:
    :return:
    """
    single = False
    if not isinstance(nodes, (tuple, list)):
        nodes = [nodes, ]
        single = True

    fused_conv_bn = True
    if "fused_conv_bn" in kwargs:
        fused_conv_bn = kwargs["fused_conv_bn"]
        del kwargs["fused_conv_bn"]

    if len(kwargs) > 0:
        sys.stderr.write("There are unrecognized option: {}\n".format(kwargs))

    ready = {}
    if fused_conv_bn:
        nodes = [fused_conv_bn_convert(o, ready=ready) for o in nodes]

    if single:
        return nodes[0]

    return nodes
