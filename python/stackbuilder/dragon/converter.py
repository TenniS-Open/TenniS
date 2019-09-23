import onnx
from onnx import numpy_helper
from onnx import optimizer

import tensorstack as ts
from ..onnx.converter import convert as convert_onnx
from ..onnx.converter import register_layer_converter as register_onnx_layer_converter
from ..onnx.converter import topy
from ..onnx.converter import convert_gemm_layer as convert_onnx_gemm_layer

import numpy


specific_layer2converter = {
}


def register_specific_layer_converter(layer, converter):
    specific_layer2converter[layer] = converter


def convert(input_file, output_file, check_graph=False):
    """
    convert onnx
    :param input_file: onnx.ModelProto or param can parse into onnx.load(param)
    :param output_file: str of path to file
    :param check_graph: if call onnx.checker.check_graph
    :return: ts.Module
    """
    return convert_onnx(input_file, output_file, check_graph, specific_layer2converter)


def convert_pooling2d_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    import tensorstack.frontend.onnx as onnx_node
    from ..onnx.converter import Name

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

    auto_pad = None
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
    ceil_mode = True    # dragon default ceil_mode = True
    if Name.Attr.ceil_mode in attr_dict:
        ceil_mode = attr_dict[Name.Attr.ceil_mode] != 0

    # if ceil_mode is not None:
    #     raise NotImplementedError("ceil_mode = {}".format(ceil_mode))

    if auto_pad is None:
        auto_pad = Name.VALID
    else:
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

    ts_node = ts.frontend.dragon.pooling2d(node_name, x=x,
                                           ksize=[1, 1, kernel_shape[0], kernel_shape[1]],
                                           stride=[1, 1, strides[0], strides[1]],
                                           type=pool_type,
                                           format=ts.zoo.Name.NCHW,
                                           padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                           padding_type=ts_padding_type,
                                           auto_pad=auto_pad,
                                           ceil_mode=ceil_mode)
    return ts_node,


register_specific_layer_converter("MaxPool", convert_pooling2d_layer)
register_specific_layer_converter("AveragePool", convert_pooling2d_layer)


def convert_gemm_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]

    assert len(input_nodes) == 3

    x = input_nodes[0]
    w = input_nodes[1]
    b = input_nodes[2]
    x = ts.zoo.flatten(name=x.name + "_flatten", x=x)

    return convert_onnx_gemm_layer(node, [x, w, b], output_names)


register_specific_layer_converter("Gemm", convert_gemm_layer)


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


register_onnx_layer_converter("ATen", convert_aten_layer)


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
