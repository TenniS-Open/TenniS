#!python
# coding: UTF-8
"""
author: kier
"""


from loadnet import load_net
from loadnet import hd
from loadnet import OPType

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
            has_header=False):
    header, net = load_net(input_file, has_header)

    if output_blobs is None:
        output_blobs = []

    if isinstance(output_blobs, tuple):
        output_blobs = list(output_blobs)
    elif not isinstance(output_blobs, list):
        output_blobs = [output_blobs, ]

    if header is not None and header.blob_name not in output_blobs:
        output_blobs.append(header.blob_name)

    if len(output_blobs) == 0:
        raise Exception("#param output_blobs must be set as having no header")

    layer_converters = {
        OPType.Enum_MemoryDataLayer: convert_memorydata_layer,
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


def convert_memorydata_layer(layer, input_nodes, output_names):
    # type: (hd.Holiday_LayerParameter, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# Converting Layer: {}".format(output_names))

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

    print("--## Input shape: {}".format(input_shape))

    input = ts.menu.param("_input", shape=input_shape)
    label = ts.menu.data(output_names[1], value=0, device=ts.device.CPU)

    input = ts.zoo.to_float("_float_input", x=input)

    scale = 1
    if param.HasField("scale"):
        scale = param.scale
        print("--## Scale: {}".format(scale))

    mean_file = None
    if param.HasField("mean_file"):
        mean_file = param.mean_file
        print("--## Mean file: {}".format(mean_file is not None))

    mean_value = list(param.mean_value)

    channel_waps = list(param.channel_swaps)

    prewhiten = False
    if param.HasField("prewhiten"):
        prewhiten = param.prewhiten
        print("--## Prewhiten: {}".format(prewhiten))

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
        input = ts.zoo.mul("_mul_scale_input", input, scale)

    if len(channel_waps) > 0:
        input = ts.zoo.dimsuffle("_channel_swap_input",
                                 input, dim=1, shuffle=numpy.asarray(channel_waps, dtype=numpy.int32))

    if prewhiten:
        input = ts.zoo.prewhiten("_prewhiten_input", input)

    input.name = output_names[0]

    return input, label


if __name__ == "__main__":

    convert("test.ext.dat", "test.tsm", output_blobs=["fc_pose_umd"], has_header=True)
