import onnx
from onnx import numpy_helper
from onnx import optimizer

import tensorstack as ts
from . import onnx_dtype as dtype
import tensorstack.frontend.onnx as onnx_node

import numpy


def to_tensor_shape(tensor_shape):
    shape = []
    for dim in tensor_shape.dim:
        shape.append(dim.dim_value)
    return shape


def get_tensor_stack_passes():
    return [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_dropout",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
        # "extract_constant_to_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        # "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_transpose_into_gemm",
        "lift_lexical_references",
        "nop",
        # "split_init",
        # "split_predict",
    ]


class Name(object):
    class Attr(object):
        group = "group"
        auto_pad = "auto_pad"
        dilations = "dilations"
        kernel_shape = "kernel_shape"
        pads = "pads"
        strides = "strides"
        storage_order = "storage_order"

        axis = "axis"
        axes = "axes"

        alpha = "alpha"
        beta = "beta"
        transA = "transA"
        transB = "transB"
        epsilon = "epsilon"

        mode = "mode"
        value = "value"

        count_include_pad = "count_include_pad"
        ceil_mode = "ceil_mode"

        output_padding = "output_padding"
        output_shape = "output_shape"

    NOTSET = "NOTSET"
    SAME_UPPER = "SAME_UPPER"
    SAME_LOWER = "SAME_LOWER"
    VALID = "VALID"

    constant = "constant"
    reflect = "reflect"
    edge = "edge"


layer2converter = {
}


def register_layer_converter(layer, converter):
    layer2converter[layer] = converter


def convert(input_file, output_file, check_graph=False):
    """
    convert onnx
    :param input_file: onnx.ModelProto or param can parse into onnx.load(param)
    :param output_file: str of path to file
    :param check_graph: if call onnx.checker.check_graph
    :return: ts.Module
    """
    onnx_model = None
    if isinstance(input_file, onnx.ModelProto):
        onnx_model = input_file
    else:
        onnx_model = onnx.load(input_file)

    if onnx_model is None:
        raise Exception("Can not load {}:{} to onnx model".format(type(input_file), input_file))

    if check_graph:
        onnx.checker.check_graph(onnx_model.graph)
    onnx_model = optimizer.optimize(onnx_model, get_tensor_stack_passes())

    onnx_graph = onnx_model.graph

    # op
    nodes = []
    print("==================== Node ====================")
    for node in onnx_graph.node:
        op_type = node.op_type
        attribute = node.attribute
        # print("{}: {} => {}".format(node.op_type, list(node.input), list(node.output)))
        # print("{}".format(attribute))
        nodes.append(node)
    print ("Got {} nodes.".format(len(nodes)))

    # init
    initialized = {}    # str: numpy.array
    print("==================== Initializer ====================")
    for tensor in onnx_graph.initializer:
        name = tensor.name
        array = numpy_helper.to_array(tensor)
        # print("{}: {}, {}".format(name, array.dtype, array.shape))
        initialized[name] = array
    print ("Got {} initializer.".format(len(initialized)))

    input = {}  # str, shape
    # input
    print("==================== Input ====================")
    for value_info in onnx_graph.input:
        name = value_info.name
        if name in initialized:
            continue
        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type
        shape = to_tensor_shape(tensor_type.shape)
        print("{}: {}, {}".format(name, elem_type, shape))
        input[name] = (elem_type, shape)

    output = {} # str, shape
    graph_output_names = []
    # output
    print("==================== Output ====================")
    for value_info in onnx_graph.output:
        name = value_info.name
        if name in initialized:
            continue
        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type
        shape = to_tensor_shape(tensor_type.shape)
        print("{}: {}, {}".format(name, elem_type, shape))
        output[name] = (elem_type, shape)
        graph_output_names.append(name)

    # set all initialized node
    name2node = {}  # str -> ts.Node
    # get ts_inputs
    ts_inputs = []
    # no loop in graph
    for name in input.keys():
        value = input[name]
        elem_type = value[0]
        shape = value[1]
        ts_dtype = dtype.from_onnx(elem_type)
        ts_input_node = ts.menu.param("_input_" + name, shape=shape)
        ts_node = ts.zoo.cast(name, x=ts_input_node, dtype=ts_dtype)
        name2node[name] = ts_node
        ts_inputs.append(ts_input_node)

    for name in initialized.keys():
        value = initialized[name]
        ts_node = ts.menu.data(name, value=value)
        name2node[name] = ts_node

    layer_converters = {
        "Conv": convert_conv_layer,
        "Relu": convert_relu_layer,
        "MaxPool": convert_pooling2d_layer,
        "Add": convert_add_layer,
        "AveragePool": convert_pooling2d_layer,
        "Shape": convert_shape_layer,
        "Concat": convert_concat_layer,
        "BatchNormalization": convert_bn_layer,
        "Pad": convert_pad_layer,
        "Constant": convert_constant_layer,
        # about new operator
        "Gather": convert_gather_layer,
        "Unsqueeze": convert_unsqueeze_layer,
        "Reshape": convert_reshape_layer,
        "Gemm": convert_gemm_layer,
        "GlobalAveragePool": convert_global_pooling2d_layer,
        "Sigmoid": convert_sigmoid_layer,
        "Neg": convert_neg_layer,
        "Transpose": convert_transpose_layer,
        "Softmax": convert_softmax_layer,
    }
    layer_converters.update(layer2converter)

    print("==================== Converting ====================")
    # convert each node
    for node in nodes:
        op_type = node.op_type
        # attribute = node.attribute
        node_input = node.input
        node_output = node.output

        # convert layer
        if op_type not in layer_converters:
            raise Exception("Not supported ONNX Layer {}".format(op_type))
        ts_converter = layer_converters[op_type]

        input_ts_nodes = []
        for name in node_input:
            input_ts_nodes.append(name2node[name])

        output_names = []
        for name in node_output:
            output_names.append(name)

        output_ts_nodes = ts_converter(node, input_ts_nodes, output_names)

        if isinstance(output_ts_nodes, ts.Node):
            output_ts_nodes = (output_ts_nodes, )

        assert len(output_names) == len(output_ts_nodes)

        for i in range(len(output_ts_nodes)):
            # update blob2nodes
            name2node[node_output[i]] = output_ts_nodes[i]

    # get outputs from outout_blobs
    ts_outputs = []
    for name in graph_output_names:
        if name not in name2node:
            raise Exception("Not computed node: {}".format(name))
        ts_outputs.append(name2node[name])

    module = ts.Module()

    # load module
    module.load(ts_outputs)

    # sort inputs
    print(ts_inputs)
    module.sort_inputs(ts_inputs)

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

    return module


def topy(attr):
    # type: (onnx.AttributeProto) -> object
    type = attr.type
    if type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    elif type == onnx.AttributeProto.STRING:
        return bytes(attr.s).decode("UTF-8")
    elif type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif type == onnx.AttributeProto.INT:
        return attr.i
    else:
        raise Exception("Can not convert attribute: {}".format(attr))


def convert_conv_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2 or len(input_nodes) == 3
    assert len(output_names) == 1

    conv2d_name = "_conv2d_" + output_names[0]
    bias_name = "_bias_" + output_names[0]
    node_name = output_names[0]

    X = input_nodes[0]
    W = input_nodes[1]  # (M x C/group x kH x kW)
    B = None
    if len(input_nodes) > 2:
        B = input_nodes[2]

    auto_pad = Name.NOTSET
    if Name.Attr.auto_pad in attr_dict:
        auto_pad = attr_dict[Name.Attr.auto_pad]
        print("--##    AutoPad: {}".format(auto_pad))

    dilations = attr_dict[Name.Attr.dilations]
    print("--##    Dilations: {}".format(dilations))

    group = 1
    if Name.Attr.group in attr_dict:
        group = attr_dict[Name.Attr.group]
        print("--##    Group: {}".format(group))

    kernel_shape = attr_dict[Name.Attr.kernel_shape]
    print("--##    KernelShape: {}".format(kernel_shape))

    pads = attr_dict[Name.Attr.pads]
    print("--##    Pads: {}".format(pads))

    strides = attr_dict[Name.Attr.strides]
    print("--##    Strides: {}".format(strides))

    if auto_pad != Name.NOTSET:
        raise NotImplementedError("auto_pad = {}".format(auto_pad))

    if len(dilations) != 2:
        raise NotImplementedError("dilations = {}".format(dilations))

    if len(kernel_shape) != 2:
        raise NotImplementedError("kernel_shape = {}".format(kernel_shape))

    W_array = ts.zoo.to_const(W, "W")

    if len(W_array.shape) != 4:
        raise NotImplementedError("W.shape = {}".format(W_array.shape))

    if group != 1 and W_array.shape[1] != 1:
        raise NotImplementedError("group = {} with weights.shape[1] = {}".format(group, W_array.shape[1]))

    if kernel_shape[0] != W_array.shape[2] or kernel_shape[1] != W_array.shape[3]:
        raise NotImplementedError("kernel_shape = {} with W.shape = {}".format(kernel_shape, W_array.shape))

    if len(pads) != 4:
        raise NotImplementedError("pads = {}".format(pads))

    if len(strides) != 2:
        raise NotImplementedError("strides = {}".format(strides))

    is_conv2d = group == 1
    is_depthwise_conv2d = W_array.shape[1] == 1

    ts_node = None

    if is_conv2d:
        ts_node = ts.zoo.conv2d(conv2d_name, x=input_nodes[0], w=W, format=ts.zoo.Name.NCHW,
                                padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                padding_value=0,
                                stride=[1, 1, strides[0], strides[1]],
                                dilation=[1, 1, dilations[0], dilations[1]])
    elif is_depthwise_conv2d:
        weights_shape = W_array.shape
        depthwise_weights_shape = (weights_shape[1], weights_shape[0], weights_shape[2], weights_shape[3])
        weights_blob = W_array.reshape(depthwise_weights_shape)
        ts_node = ts.zoo.depthwise_conv2d(conv2d_name, x=input_nodes[0], w=weights_blob, format=ts.zoo.Name.NCHW,
                                          padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                          padding_value=0,
                                          stride=[1, 1, strides[0], strides[1]],
                                          dilation=[1, 1, dilations[0], dilations[1]])

    if ts_node is None:
        raise NotImplementedError(node)

    if B is not None:
        ts_node = ts.zoo.add_bias(bias_name, x=ts_node, b=B, format=ts.zoo.Name.NCHW)

    ts_node.name = node_name

    return ts_node,


def convert_relu_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.relu(node_name, x=x)

    return ts_node,


def convert_neg_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.sub(name=node_name, lhs=numpy.asarray(0, dtype=numpy.float32), rhs=x)

    return ts_node,


def convert_pooling2d_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    op_type = node.op_type
    onnx_op_type_to_ts_pool_type = {
        "MaxPool": ts.zoo.Type.pooling_type.max,
        "AveragePool": ts.zoo.Type.pooling_type.avg,
    }

    auto_pad = Name.NOTSET
    if Name.Attr.auto_pad in attr_dict:
        auto_pad = attr_dict[Name.Attr.auto_pad]
        print("--##    AutoPad: {}".format(auto_pad))

    kernel_shape = attr_dict[Name.Attr.kernel_shape]
    print("--##    KernelShape: {}".format(kernel_shape))

    pads = attr_dict[Name.Attr.pads]
    print("--##    Pads: {}".format(pads))

    storage_order = 0
    if Name.Attr.storage_order in attr_dict:
        storage_order = attr_dict[Name.Attr.storage_order]
        print("--##    StorageOrder: {}".format(storage_order))

    strides = attr_dict[Name.Attr.strides]
    print("--##    Strides: {}".format(strides))

    count_include_pad = False
    if Name.Attr.count_include_pad in attr_dict:
        count_include_pad = attr_dict[Name.Attr.count_include_pad] != 0
    ceil_mode = False
    if Name.Attr.ceil_mode in attr_dict:
        ceil_mode = attr_dict[Name.Attr.ceil_mode] != 0

    if auto_pad != Name.NOTSET:
        raise NotImplementedError("auto_pad = {}".format(auto_pad))

    if storage_order != 0:
        raise NotImplementedError("storage_order = {}".format(storage_order))

    if len(kernel_shape) != 2:
        raise NotImplementedError("kernel_shape = {}".format(kernel_shape))

    if len(pads) != 4:
        raise NotImplementedError("pads = {}".format(pads))

    if len(strides) != 2:
        raise NotImplementedError("strides = {}".format(strides))

    if op_type not in onnx_op_type_to_ts_pool_type:
        raise NotImplementedError("pooling type = {}".format(op_type))
    pool_type = onnx_op_type_to_ts_pool_type[op_type]

    ts_padding_type = ts.zoo.Type.padding_type.black
    if count_include_pad:
        ts_padding_type = ts.zoo.Type.padding_type.white

    ts_node = onnx_node.pooling2d(node_name, x=x,
                                  ksize=[1, 1, kernel_shape[0], kernel_shape[1]],
                                  stride=[1, 1, strides[0], strides[1]],
                                  type=pool_type,
                                  format=ts.zoo.Name.NCHW,
                                  padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                  padding_type=ts_padding_type,
                                  auto_pad=auto_pad,
                                  ceil_mode=ceil_mode)
    return ts_node,


def convert_add_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]

    ts_node = ts.zoo.add(node_name, lhs=x, rhs=y)

    return ts_node,


def convert_shape_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.shape("_int32_" + node_name, x=x)
    ts_node = ts.zoo.cast(node_name, ts_node, dtype=ts.dtype.INT64);

    return ts_node,


def convert_gather_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    indices = input_nodes[1]

    axis = 0
    if Name.Attr.axis in attr_dict:
        axis = attr_dict[Name.Attr.axis]
        print("--##    axis: {}".format(axis))

    ts_node = onnx_node.gather(node_name, x=x, indices=indices, axis=axis)

    return ts_node,


def convert_unsqueeze_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axes = attr_dict[Name.Attr.axes]
    print("--##    axes: {}".format(axes))

    ts_node = onnx_node.unsqueeze(node_name, x=x, axes=axes)

    return ts_node,


def convert_concat_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(output_names) == 1

    node_name = output_names[0]
    print("--##    input number: {}".format(len(input_nodes)))

    axis = attr_dict[Name.Attr.axis]
    print("--##    axis: {}".format(axis))

    ts_node = ts.zoo.concat(node_name, inputs=input_nodes, dim=axis)

    return ts_node,


def __whose_flatten_shape(shape):
    # type: (ts.Node) -> Union[ts.Node, None]
    """
    :return: return flatten tensor if it's flatten shape like(x.number, -1)
    """
    if not isinstance(shape, ts.Node):
        return None

    if shape.op != ts.zoo.Name.Layer.concat:
        return None

    unsqueeze_x_number = shape.inputs[0]
    unsqueeze_neg_one = shape.inputs[1]

    if unsqueeze_x_number.op != onnx_node.Name.Layer.unsqueeze:
        return None

    if unsqueeze_neg_one.op != onnx_node.Name.Layer.unsqueeze:
        return None

    if list(unsqueeze_x_number.get(onnx_node.Name.axes)) != [0]:
        return None

    if list(unsqueeze_neg_one.get(onnx_node.Name.axes)) != [0]:
        return None

    neg_one = unsqueeze_neg_one.inputs[0]
    x_number = unsqueeze_x_number.inputs[0]

    if neg_one.op != ts.Node.Const:
        return None
    elif int(neg_one.get(ts.menu.Name.value)) != -1:
        return None

    if x_number.op != onnx_node.Name.Layer.gather:
        return None
    elif int(x_number.get(onnx_node.Name.axis)) != 0:
        return None

    x_shape = x_number.inputs[0]

    if x_shape.op == ts.zoo.Name.Layer.cast:
        x_shape = x_shape.inputs[0]

    if x_shape.op != ts.zoo.Name.Layer.shape:
        return None

    x = x_shape.inputs[0]

    return x


def convert_reshape_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    new_shape = input_nodes[1]

    flatten_x = __whose_flatten_shape(new_shape)

    if x == flatten_x:
        print("--##    IsFlatten: {}".format(True))
        return ts.zoo.flatten(node_name, x)

    ts_node = ts.zoo.reshape(node_name, x, new_shape)

    return ts_node,


def convert_gemm_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 3
    assert len(output_names) == 1

    node_name = output_names[0]

    A = input_nodes[0]
    B = input_nodes[1]
    C = input_nodes[2]

    alpha = 1.0
    if Name.Attr.alpha in attr_dict:
        alpha = attr_dict[Name.Attr.alpha]
    print("--##    alpha: {}".format(alpha))

    beta = 1.0
    if Name.Attr.beta in attr_dict:
        beta = attr_dict[Name.Attr.beta]
    print("--##    beta: {}".format(beta))

    transA = 0
    if Name.Attr.transA in attr_dict:
        transA = attr_dict[Name.Attr.transA]
    print("--##    transA: {}".format(transA))

    transB = 0
    if Name.Attr.transB in attr_dict:
        transB = attr_dict[Name.Attr.transB]
    print("--##    transB: {}".format(transB))

    ts_node = onnx_node.gemm(node_name, A=A, B=B, C=C, alpha=alpha, beta=beta, transA=transA, transB=transB)

    return ts_node,


def convert_bn_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 5
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    scale = input_nodes[1]
    B = input_nodes[2]
    mean = input_nodes[3]
    var = input_nodes[4]

    epsilon = 1e-5
    if Name.Attr.epsilon in attr_dict:
        epsilon = attr_dict[Name.Attr.epsilon]
        print("--##    epsilon: {}".format(epsilon))

    ts_node = ts.zoo.fused_batch_norm(node_name, x=x,
                                      mean=mean, variance=var, scale=scale, bias=B,
                                      dim=1, epsilon=epsilon)

    return ts_node,


def convert_pad_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    mode = Name.constant
    if Name.Attr.mode in attr_dict:
        mode = attr_dict[Name.Attr.mode]
        print("--##    mode: {}".format(mode))

    if mode != Name.constant:
        raise NotImplementedError("mode={}".format(mode))

    pads = attr_dict[Name.Attr.pads]
    print("--##    pads: {}".format(pads))

    value = 0
    if Name.Attr.value in attr_dict:
        value = attr_dict[Name.Attr.value]
        print("--##    value: {}".format(value))

    pads = numpy.asarray(pads, dtype=numpy.int32).reshape((2, -1)).T

    ts_node = ts.zoo.pad(node_name, x=x, padding=pads, padding_value=value)

    return ts_node,


def convert_constant_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 0
    assert len(output_names) == 1

    node_name = output_names[0]

    value = attr_dict[Name.Attr.value]

    ts_node = ts.menu.data(node_name, value=value)

    return ts_node,


def convert_global_pooling2d_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    op_type = node.op_type
    onnx_op_type_to_ts_pool_type = {
        "GlobalMaxPool": ts.zoo.Type.pooling_type.max,
        "GlobalAveragePool": ts.zoo.Type.pooling_type.avg,
    }

    if op_type not in onnx_op_type_to_ts_pool_type:
        raise NotImplementedError("pooling type = {}".format(op_type))
    pool_type = onnx_op_type_to_ts_pool_type[op_type]

    ts_node = ts.zoo.global_pooling2d(node_name, x=x, type=pool_type, format=ts.zoo.Name.NCHW)

    return ts_node,


def convert_sigmoid_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.sigmoid(node_name, x=x)

    return ts_node,


def convert_transpose_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.transpose(name=node_name, x=x, permute=attr_dict["perm"])

    return ts_node,


def convert_softmax_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axis = 1
    if Name.Attr.axis in attr_dict:
        axis = int(attr_dict[Name.Attr.axis])

    ts_node = ts.zoo.softmax(name=node_name, x=x, dim=axis)

    return ts_node,


def convert_sub_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]

    ts_node = ts.zoo.sub(node_name, lhs=x, rhs=y)

    return ts_node,


register_layer_converter("Sub", convert_sub_layer)


def convert_div_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]

    ts_node = ts.zoo.div(node_name, lhs=x, rhs=y)

    return ts_node,


register_layer_converter("Div", convert_div_layer)


def convert_mul_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]

    ts_node = ts.zoo.mul(node_name, lhs=x, rhs=y)

    return ts_node,


register_layer_converter("Mul", convert_mul_layer)


def convert_reduce_sum_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axes = attr_dict["axes"]
    keepdims = bool(attr_dict["keepdims"])

    if len(axes) == 0:
        node = ts.zoo.reshape(node_name + "_flatten", x=x, shape=[-1])
        ts_node = ts.zoo.reduce_sum(node_name, x=node, reduce_dims=0, keep_dims=keepdims)
    elif len(axes) == 1:
        ts_node = ts.zoo.reduce_sum(node_name, x=x, reduce_dims=axes[0], keep_dims=keepdims)
    else:
        raise NotImplementedError("axes = {}".format(axes))

    return ts_node,


register_layer_converter("ReduceSum", convert_reduce_sum_layer)


def convert_sqrt_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.sqrt(node_name, x)

    return ts_node,


register_layer_converter("Sqrt", convert_sqrt_layer)


def convert_flatten_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axis = 1
    if "axis" in attr_dict:
        axis = attr_dict["axis"]

    ts_node = ts.zoo.flatten(node_name, x=x, dim=axis)

    return ts_node,


register_layer_converter("Flatten", convert_flatten_layer)


def convert_conv_traspose_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2 or len(input_nodes) == 3
    assert len(output_names) == 1

    conv2d_name = "_conv2d_" + output_names[0]
    bias_name = "_bias_" + output_names[0]
    node_name = output_names[0]

    X = input_nodes[0]
    W = input_nodes[1]  # (M x C/group x kH x kW)
    B = None
    if len(input_nodes) > 2:
        B = input_nodes[2]

    auto_pad = Name.NOTSET
    if Name.Attr.auto_pad in attr_dict:
        auto_pad = attr_dict[Name.Attr.auto_pad]
        print("--##    AutoPad: {}".format(auto_pad))

    dilations = attr_dict[Name.Attr.dilations]
    print("--##    Dilations: {}".format(dilations))

    group = 1
    if Name.Attr.group in attr_dict:
        group = attr_dict[Name.Attr.group]
        print("--##    Group: {}".format(group))

    kernel_shape = attr_dict[Name.Attr.kernel_shape]
    print("--##    KernelShape: {}".format(kernel_shape))

    pads = attr_dict[Name.Attr.pads]
    print("--##    Pads: {}".format(pads))

    strides = attr_dict[Name.Attr.strides]
    print("--##    Strides: {}".format(strides))

    output_padding = None
    if Name.Attr.output_padding in attr_dict:
        output_padding = attr_dict[Name.Attr.output_padding]
        print("--##    output_padding: {}".format(output_padding))

    output_shape = None
    if Name.Attr.output_shape in attr_dict:
        output_shape = attr_dict[Name.Attr.output_shape]
        print("--##    output_shape: {}".format(output_shape))

    if output_shape is not None:
        raise NotImplementedError("output_shape = {}".format(output_shape))

    if auto_pad != Name.NOTSET:
        raise NotImplementedError("auto_pad = {}".format(auto_pad))

    if group != 1:
        raise NotImplementedError("group = {}".format(group))

    if len(dilations) != 2:
        raise NotImplementedError("dilations = {}".format(dilations))

    if len(kernel_shape) != 2:
        raise NotImplementedError("kernel_shape = {}".format(kernel_shape))

    W_array = ts.zoo.to_const(W, "W")

    if len(W_array.shape) != 4:
        raise NotImplementedError("W.shape = {}".format(W_array.shape))

    if group != 1 and W_array.shape[1] != 1:
        raise NotImplementedError("group = {} with weights.shape[1] = {}".format(group, W_array.shape[1]))

    if kernel_shape[0] != W_array.shape[2] or kernel_shape[1] != W_array.shape[3]:
        raise NotImplementedError("kernel_shape = {} with W.shape = {}".format(kernel_shape, W_array.shape))

    if len(pads) != 4:
        raise NotImplementedError("pads = {}".format(pads))

    if len(strides) != 2:
        raise NotImplementedError("strides = {}".format(strides))

    is_conv2d = group == 1
    # is_depthwise_conv2d = W_array.shape[1] == 1

    ts_node = None

    if is_conv2d:
        ts_node = ts.zoo.transpose_conv2d(conv2d_name, x=input_nodes[0], w=W, format=ts.zoo.Name.NCHW,
                                          padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                          padding_value=0,
                                          stride=[1, 1, strides[0], strides[1]],
                                          dilation=[1, 1, dilations[0], dilations[1]])

    if output_padding is not None:
        assert len(output_padding) == 4
        ts_node = ts.zoo.pad(node_name + "_out_pad", x=ts_node,
                             padding=[[0, 0], [0, 0], [output_padding[0], output_padding[2]], [output_padding[1], output_padding[3]]])

    if ts_node is None:
        raise NotImplementedError(node)

    if B is not None:
        ts_node = ts.zoo.add_bias(bias_name, x=ts_node, b=B, format=ts.zoo.Name.NCHW)

    ts_node.name = node_name

    return ts_node,


register_layer_converter("ConvTranspose", convert_conv_traspose_layer)


def convert_tile_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    repeats = input_nodes[1]

    ts_node = ts.zoo.tile(node_name, x=x, repeats=repeats)

    return ts_node,


register_layer_converter("Tile", convert_tile_layer)


def convert_dropout_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.copy(node_name, x=x)

    return ts_node,


register_layer_converter("Dropout", convert_dropout_layer)


def convert_tanh_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.tanh(node_name, x=x)

    return ts_node,


register_layer_converter("Tanh", convert_tanh_layer)


def convert_abs_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.abs(node_name, x=x)

    return ts_node,


register_layer_converter("Abs", convert_abs_layer)


aten_layer2converter = {
}


def register_aten_layer_converter(layer, converter):
    aten_layer2converter[layer] = converter


def convert_aten_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    op_type = attr_dict["op_type"]

    if op_type not in aten_layer2converter:
        raise Exception("Not supported ATen Layer {}".format(op_type))
    ts_converter = aten_layer2converter[op_type]

    return ts_converter(node, input_nodes, output_names)


register_layer_converter("ATen", convert_aten_layer)


def convert_image_data_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    x = ts.zoo.to_float(node_name + "_float", x)

    if "mean_values" in attr_dict:
        mean_values = attr_dict["mean_values"]
        mean_values = numpy.reshape(numpy.asarray(mean_values, dtype=numpy.float32), (1, 1, 1, -1))
        x = ts.zoo.sub(node_name + "_sub_mean", x, mean_values, dtype=numpy.float32)

    if "std_values" in attr_dict:
        std_values = attr_dict["std_values"]
        std_values = numpy.reshape(numpy.asarray(std_values, dtype=numpy.float32), (1, 1, 1, -1))
        std_values = 1. / std_values
        x = ts.zoo.mul(node_name + "_div_std", x, std_values, dtype=numpy.float32)

    data_format = attr_dict["data_format"]
    if data_format == "NCHW":
        x = ts.zoo.transpose(node_name + "_nchw", x, (0, 3, 1, 2))
    elif data_format == "NHWC":
        pass
    else:
        raise NotImplementedError("data_format={}".format(data_format))

    node = x
    node.name = node_name

    return node,


register_aten_layer_converter("ImageData", convert_image_data_layer)


def convert_affine_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 3
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    A = input_nodes[1]
    b = input_nodes[2]

    axis = attr_dict["axis"]
    num_axes = attr_dict["num_axes"]

    node = None
    if num_axes == 1:
        node = ts.zoo.batch_scale(node_name, x=x, scale=A, bias=b, dim=axis)
    else:
        A_value = ts.zoo.to_const(A, "A")
        b_value = ts.zoo.to_const(b, "b")
        param_shape = list(A_value.shape)

        if len(param_shape) != num_axes:
            raise NotImplementedError("num_axes={}, A.shape={}, b.shape={}".
                                      format(num_axes, A_value.shape, b_value.shape))

        for i in range(int(axis)):
            param_shape.insert(0, 1)
        while len(param_shape) < 4:
            param_shape.append(1)
        if len(param_shape) > 4:
            raise NotImplementedError("num_axes={}, A.shape={}, b.shape={}".
                                      format(num_axes, A_value.shape, b_value.shape))
        broadcast_A = numpy.reshape(A_value, param_shape)
        broadcast_b = numpy.reshape(b_value, param_shape)

        broadcast_A = ts.menu.data(A.name + "_broadcast", broadcast_A)
        broadcast_b = ts.menu.data(b.name + "_broadcast", broadcast_b)

        Ax = ts.zoo.mul(name=node_name + "Ax", lhs=x, rhs=broadcast_A, dtype=numpy.float32)
        Ax_b = ts.zoo.add(name=node_name + "Ax_b", lhs=Ax, rhs=broadcast_b, dtype=numpy.float32)

        node = Ax_b
        node.name = node_name

    return node,


register_aten_layer_converter("Affine", convert_affine_layer)


def convert_upsample_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    scales = input_nodes[1]

    scale = ts.zoo.to_const(scales, "scales")

    mode = attr_dict["mode"]
    mode2type = {
        "nearest": ts.zoo.Type.resize2d_type.nearest,
        "bilinear": ts.zoo.Type.resize2d_type.linear,
    }
    if mode not in mode2type:
        raise NotImplementedError("mode={}".format(mode))
    type = mode2type[mode]

    ts_node = ts.zoo.sample2d(name=node_name, x=x, scale=scale, type=type)

    return ts_node,


register_layer_converter("Upsample", convert_upsample_layer)


def convert_proposal_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) >= 3
    assert len(output_names) >= 1

    inputs = input_nodes

    proposals = ts.frontend.dragon.proposal(
        output_names=output_names,
        inputs=inputs,
        strides=attr_dict["strides"],
        ratios=attr_dict["ratios"],
        scales=attr_dict["scales"],
        pre_nms_top_n=attr_dict["pre_nms_top_n"],
        post_nms_top_n=attr_dict["post_nms_top_n"],
        nms_thresh=attr_dict["nms_thresh"],
        min_size=attr_dict["min_size"],
        min_level=attr_dict["min_level"],
        max_level=attr_dict["max_level"],
        canonical_scale=attr_dict["canonical_scale"],
        canonical_level=attr_dict["canonical_level"],
    )

    return proposals


register_aten_layer_converter("Proposal", convert_proposal_layer)


def convert_roi_align_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    inputs = input_nodes

    regions = ts.frontend.dragon.roi_align(
        output_names=output_names,
        inputs=inputs,
        pool_h=attr_dict["pool_h"],
        pool_w=attr_dict["pool_w"],
        spatial_scale=attr_dict["spatial_scale"],
        sampling_ratio=attr_dict["sampling_ratio"],
    )

    return regions


register_aten_layer_converter("ROIAlign", convert_roi_align_layer)
