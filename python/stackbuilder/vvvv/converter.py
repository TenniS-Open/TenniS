"""
@author: kier
"""

from . import net
import tensorstack as ts


def convert(input_model, output_file, input_num=None, inputs=None):
    """
    convert vvvv model to tsm
    :param input_model: str or IOStream, contain exact vvvv model
    :param output_file: str of path to file
    :param input_num: input number
    :param inputs: input nodes
    :return: ts.Module
    """
    input_model = net.compatible_string(input_model)

    module = None
    if isinstance(input_model, str):
        with open(input_model, "rb") as fi:
            module = net.convert(fi, input_num=input_num, inputs=inputs)
    else:
        module = net.convert(input_model, input_num=input_num, inputs=inputs)

    with open(output_file, "wb") as fo:
        ts.Module.Save(stream=fo, module=module)

    print("============ Summary ============")
    if isinstance(input_model, str):
        print("Input file: {}".format(input_model))
    else:
        print("Input model from memory")
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
