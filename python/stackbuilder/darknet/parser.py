from .layers import *

from .enum import *
from .config import *

from .darknet import is_network
from .darknet import make_network
from .darknet import SizeParams
from .darknet import fprintf
from .darknet import get_network_output_layer

import sys

def string_to_layer_type(type):
    if strcmp(type, "[shortcut]"): return SHORTCUT
    if strcmp(type, "[crop]"): return CROP
    if strcmp(type, "[cost]"): return COST
    if strcmp(type, "[detection]"): return DETECTION
    if strcmp(type, "[region]"): return REGION
    if strcmp(type, "[yolo]"): return YOLO
    if strcmp(type, "[iseg]"): return ISEG
    if strcmp(type, "[local]"): return LOCAL
    if strcmp(type, "[conv]") or strcmp(type, "[convolutional]"): return CONVOLUTIONAL
    if strcmp(type, "[deconv]") or strcmp(type, "[deconvolutional]"): return DECONVOLUTIONAL
    if strcmp(type, "[activation]"): return ACTIVE
    if strcmp(type, "[logistic]"): return LOGXENT
    if strcmp(type, "[l2norm]"): return L2NORM
    if strcmp(type, "[net]") or strcmp(type, "[network]"): return NETWORK
    if strcmp(type, "[crnn]"): return CRNN
    if strcmp(type, "[gru]"): return GRU
    if strcmp(type, "[lstm]"): return LSTM
    if strcmp(type, "[rnn]"): return RNN
    if strcmp(type, "[conn]") or strcmp(type, "[connected]"): return CONNECTED
    if strcmp(type, "[max]") or strcmp(type, "[maxpool]"): return MAXPOOL
    if strcmp(type, "[reorg]"): return REORG
    if strcmp(type, "[avg]") or strcmp(type, "[avgpool]"): return AVGPOOL
    if strcmp(type, "[dropout]"): return DROPOUT
    if strcmp(type, "[lrn]") or strcmp(type, "[normalization]"): return NORMALIZATION
    if strcmp(type, "[batchnorm]"): return BATCHNORM
    if strcmp(type, "[soft]") or strcmp(type, "[softmax]"): return SOFTMAX
    if strcmp(type, "[route]"): return ROUTE
    if strcmp(type, "[upsample]"): return UPSAMPLE
    return BLANK


map_layer_parsers = {
    CONVOLUTIONAL: parse_convolutional,
    MAXPOOL: parse_max_pool,
    YOLO: parse_yolo,
    ROUTE: parse_route,
    UPSAMPLE: parse_upsample,
    SHORTCUT: parse_shortcut,
}


def parse_layer(options, params):
    # type: (Session, SizeParams) -> Layer
    layer_type = string_to_layer_type(options.name)
    if layer_type not in map_layer_parsers:
        raise NotImplementedError("Not support layer: {}".format(options.name))
    # parse base
    layer = map_layer_parsers[layer_type](options, params)
    # parse common
    l = layer
    net = params.net
    l.clip = net.clip
    l.truth = option_find_int_quiet(options, "truth", 0)
    l.onlyforward = option_find_int_quiet(options, "onlyforward", 0)
    l.stopbackward = option_find_int_quiet(options, "stopbackward", 0)
    l.dontsave = option_find_int_quiet(options, "dontsave", 0)
    l.dontload = option_find_int_quiet(options, "dontload", 0)
    l.numload = option_find_int_quiet(options, "numload", 0)
    l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0)
    l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1)
    l.smooth = option_find_float_quiet(options, "smooth", 0)

    # additional
    l.type_string = options.name

    return layer


def parse_network_cfg(filename):
    # type: (str) -> Network
    """
    :param filename: cfg file
    :return: Network
    """

    # read config
    cfg = read_config(filename)

    if len(cfg) == 0:
        raise Exception("Config file has no sections")

    net_options = cfg[0]

    if not is_network(net_options):
        raise Exception("First section must be [net] or [network]")

    net = make_network(len(cfg) - 1, net_options)

    params = SizeParams(net)

    max_count = len(cfg) - 1
    count = 0

    fprintf(sys.stderr, "layer     filters    size              input                output\n")
    while count < max_count:
        params.index = count
        options = cfg[count + 1]

        fprintf(sys.stderr, "%5d ", count)

        l = parse_layer(options, params)

        net.layers[count] = l
        if count + 1 < max_count:
            params.h = l.out_h
            params.w = l.out_w
            params.c = l.out_c
            params.inputs = l.outputs

        count += 1

    out = get_network_output_layer(net)

    net.outputs = out.outputs
    net.truths = out.outputs
    if net.layers[net.n-1].has("truths"):
        net.truths = net.layers[net.n-1].truths
    net.output = out.output
    net.input = None    # calloc(net->inputs*net->batch, sizeof(float))
    net.truth = None    # calloc(net->truths*net->batch, sizeof(float))

    return net