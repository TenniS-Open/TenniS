import onnx
from onnx import numpy_helper
from onnx import optimizer


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
        "extract_constant_to_initializer",
        # "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        # "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        # "fuse_transpose_into_gemm",
        "lift_lexical_references",
        "nop",
        # "split_init",
        # "split_predict",
    ]


def convert(input_file, output_file):
    onnx_model = onnx.load(input_file)
    onnx.checker.check_graph(onnx_model.graph)

    onnx_model = optimizer.optimize(onnx_model, get_tensor_stack_passes())

    onnx_graph = onnx_model.graph

    # op
    print("==================== Node ====================")
    for node in onnx_graph.node:
        op_type = node.op_type
        attribute = node.attribute
        print("{}: {} => {}".format(node.op_type, list(node.input), list(node.output)))
        print("{}".format(attribute))

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
        input[name] = shape

    output = {} # str, shape
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
        output[name] = shape

    #