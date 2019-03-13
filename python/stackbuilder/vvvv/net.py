"""
@author: kier
"""

import sys
import struct
import io
import numpy

import tensorstack as ts
from . import param


def compatible_string(obj):
    # type: (object) -> object
    if sys.version > '3':
        pass
    else:
        if isinstance(obj, unicode):
            return str(obj)
    return obj


class HyperParam(object):
    PARAM_INT = 1
    PARAM_FLOAT = 2
    PARAM_STRING = 3

    PARAM_END = "end"

    @classmethod
    def Load(cls, stream):
        hyper_param = {}
        param_name = param.String(stream=stream)
        while param_name != cls.PARAM_END:
            type = param.Int(stream)
            if type == cls.PARAM_INT:
                hyper_param[param_name] = param.Int(stream=stream)
            elif type == cls.PARAM_FLOAT:
                hyper_param[param_name] = param.Float(stream=stream)
            elif type == cls.PARAM_STRING:
                hyper_param[param_name] = param.String(stream=stream)
            param_name = param.String(stream=stream)
        return hyper_param

    num_subnet = "num_subnet"
    num_in = "num_in"
    num_out = "num_out"


class Blob(object):
    @classmethod
    def Load(cls, stream):
        shape = param.read_param(stream, param.Int, param.Int, param.Int, param.Int)
        count = shape[0] * shape[1] * shape[2] * shape[3]

        buffer = stream.read(count * 4)
        dtype_numpy = numpy.dtype(numpy.float32)
        dtype_numpy = dtype_numpy.newbyteorder('<')
        tensor = numpy.frombuffer(buffer, dtype=dtype_numpy)
        tensor = numpy.resize(tensor, new_shape=shape)

        return tensor


GlobalNetCreatorMap = {}


class Net(object):
    NetCreatorMap = GlobalNetCreatorMap
    """
    Call construction, then set_hyper_params, setup, set_params, bind(create node), <set input and output plug>, link
    """
    def __init__(self, type=None):
        # type: (str) -> None
        self.__net_count = None
        self.__input_count = None
        self.__output_count = None
        self.__param_count = None
        self.__nets = []
        self.__inputs = []
        self.__outputs = []
        self.__params = {}

        self.__hyper_params = {}
        self.__plugs = []

        self.__type = type
        if self.__type is None:
            self.__type = self.__class__.__name__

    def set_hyper_params(self, hyper_params):
        self.__hyper_params = hyper_params

    def set_params(self, params):
        self.__params = params

    def set_plugs(self, plugs):
        self.__plugs = plugs

    def _init(self, net_count, input_count, output_count, param_count):
        # type: (int, int, int, int) -> None
        """
        init struct
        :param net_count:
        :param input_count:
        :param output_count:
        :param param_count:
        :return:
        """
        self.__net_count = net_count
        self.__input_count = input_count
        self.__output_count = output_count
        self.__param_count = param_count
        self.__nets = [None for _ in range(net_count)]
        self.__inputs = [None for _ in range(input_count)]
        self.__outputs = [None for _ in range(output_count)]

    def load(self, stream):
        # type: (file) -> None
        # load every params, the name layer has already parse
        self.__hyper_params = HyperParam.Load(stream=stream)

        # call setup, to get param count
        self.setup()

        self.__params = []
        for _ in range(self.param_count):
            self.__params.append(Blob.Load(stream=stream))

        self.__nets = []
        for _ in range(self.net_count):
            self.__nets.append(self.Load(stream))

        self.__plugs = []

        if len(self.__nets) == 0:
            return

        # for input
        for net in self.__nets:
            assert isinstance(net, Net)
            plug_list = []
            for _ in range(net.input_count):
                plug = param.read_param(stream, param.Int, param.Int)
                plug_list.append(plug)
            self.__plugs.append(plug_list)
        # for output
        plug_list = []
        for _ in range(self.output_count):
            plug = param.read_param(stream, param.Int, param.Int)
            plug_list.append(plug)
        self.__plugs.append(plug_list)

    @classmethod
    def Load(cls, stream, inputs=None):
        # type: (Any) -> Net
        net_type = param.read_param(stream, param.String)[0]

        if net_type not in cls.NetCreatorMap:
            raise NotImplementedError("\"{}\"".format(net_type))

        net = cls.NetCreatorMap[net_type]()
        net.load(stream=stream)

        if inputs != None:
            if len(inputs) != len(net.__inputs):
                raise Exception("inputs count sould be {}".format(len(inputs)))
            net.__inputs = inputs

        return net

    def setup(self):
        """
        Set net_count, input_count, output_count, param_count, means call _init in setup
        :return: None
        """
        raise NotImplementedError("{} not override setup".format(self.__class__.__name__))

    def bind(self):
        """
        create node in ts.Node
        :return: None
        """
        raise NotImplementedError("{} not override bind".format(self.__class__.__name__))

    def link(self):
        """
        Link each node by input or output plg
        :return:
        """
        raise NotImplementedError("{} not override link".format(self.__class__.__name__))

    @property
    def net_count(self):
        if self.__net_count is None:
            raise Exception("Miss-calling setup, all setup invalid")
        return self.__net_count

    @property
    def input_count(self):
        if self.__input_count is None:
            raise Exception("Miss-calling setup, all setup invalid")
        return self.__input_count

    @property
    def output_count(self):
        if self.__output_count is None:
            raise Exception("Miss-calling setup, all setup invalid")
        return self.__output_count

    @property
    def param_count(self):
        if self.__param_count is None:
            raise Exception("Miss-calling setup, all setup invalid")
        return self.__param_count

    @property
    def inputs(self):
        return self.__inputs

    @property
    def hyper_params(self):
        return self.__hyper_params

    @property
    def params(self):
        return self.__params

    @property
    def outputs(self):
        return self.__outputs

    @property
    def nets(self):
        return self.__nets

    @property
    def plugs(self):
        return self.__plugs

    def to_string(self, indent):
        next_indent = indent + "    "
        lines = []

        param_shape = ", ".join([str(param.shape) for param in self.params])
        lines.append("{}{} attr:{} param:[{}]".format(indent, self.__type, self.hyper_params, param_shape))
        for net in self.__nets:
            assert isinstance(net, Net)
            lines.append("{}".format(net.to_string(next_indent)))

        return "\n".join(lines)

    def __str__(self):
        return self.to_string("")

    def __repr__(self):
        return self.__str__()


class CommonNet(Net):
    def __init__(self):
        super(CommonNet, self).__init__()

    def setup(self):
        self._init(
            self.hyper_params[HyperParam.num_subnet],
            self.hyper_params[HyperParam.num_in],
            self.hyper_params[HyperParam.num_out],
            0)

    def bind(self):
        # no new node to create
        for net in self.nets:
            net.bind()

    def link(self):
        # link by plugs

        linked = [False] * self.net_count

        depth = 0
        max_depth = self.net_count * 2
        def link_sub_net(i):
            if linked[i]:
                raise Exception("What a Terrible Failure!")

            net = self.nets[i]
            assert isinstance(net, Net)
            for j in range(net.input_count):
                plug = self.plugs[i][j]
                net_i = plug[0]
                blob_i = plug[1]
                if net_i < 0:
                    net.inputs[j] = self.inputs[blob_i]
                else:
                    net_i_output_i = self.nets[net_i].outputs[blob_i]
                    if net_i_output_i is None or not linked[i]:
                        link_sub_net(net_i)
                    net_i_output_i = self.nets[net_i].outputs[blob_i]
                    assert net_i_output_i is not None
                    net.inputs[j] = net_i_output_i
                for input in net.inputs:
                    assert isinstance(input, ts.Node)
                net.link()
                linked[i] = True

        output_plug = self.plugs[self.net_count]
        for j in range(self.output_count):
            plug = output_plug[j]
            net_i = plug[0]
            blob_i = plug[1]
            if net_i < 0:
                self.outputs[j] = self.inputs[blob_i]
            else:
                if not linked[net_i]:
                    link_sub_net(net_i)
                    linked[net_i] = True
                self.outputs[j] = self.nets[net_i].outputs[blob_i]

        for output in self.outputs:
            assert isinstance(output, ts.Node)


GlobalNetCreatorMap["Common"] = CommonNet


class ConvNet(Net):
    def __init__(self):
        super(ConvNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=1)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["Conv"] = ConvNet


class BiasAdderNet(Net):
    def __init__(self):
        super(BiasAdderNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=1)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["BiasAdder"] = BiasAdderNet


class BnNet(Net):
    def __init__(self):
        super(BnNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=3)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["Bn"] = BnNet


class ScaleNet(Net):
    def __init__(self):
        super(ScaleNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=2)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["Scale"] = ScaleNet


class ReLUNet(Net):
    def __init__(self):
        super(ReLUNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=0)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["ReLU"] = ReLUNet


class MaxPoolingNet(Net):
    def __init__(self):
        super(MaxPoolingNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=0)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["MaxPooling"] = MaxPoolingNet


class PadNet(Net):
    def __init__(self):
        super(PadNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=0)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["Pad"] = PadNet


class EltwiseOPNet(Net):
    def __init__(self):
        super(EltwiseOPNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=0)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["EltwiseOP"] = EltwiseOPNet


class InnerProductNet(Net):
    def __init__(self):
        super(InnerProductNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=1, output_count=1, param_count=1)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["InnerProduct"] = InnerProductNet


class ShapeIndexPatchNet(Net):
    def __init__(self):
        super(ShapeIndexPatchNet, self).__init__()

    def setup(self):
        self._init(net_count=0, input_count=2, output_count=1, param_count=0)

    def bind(self):
        self.outputs[0] = ts.Node(op="fake", name=self.__class__.__name__, output_count=1)

    def link(self):
        ts.Node.Link(self.outputs[0], self.inputs)


GlobalNetCreatorMap["ShapeIndexPatch"] = ShapeIndexPatchNet


def convert(stream, input_num=1):
    inputs = [ts.menu.param("_input_" + str(i)) for i in range(input_num)]
    float_inputs = [None] * len(inputs)
    for i in range(input_num):
        float_inputs[i] = ts.zoo.to_float(name="_float_input_" + str(i), x=inputs[i])
    net = Net.Load(stream=stream, inputs=inputs)

    print("============ Loaded vvvv model ============")
    print(net)

    print("============ Converting ============")
    net.bind()
    net.link()

    module = ts.Module()

    # load module
    module.load(net.outputs)

    module.sort_inputs(inputs)

    return module