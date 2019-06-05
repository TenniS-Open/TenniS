#!python
# coding: UTF-8
"""
author: kier
"""

from .config import Session
from .config import option_find_int
from .config import option_find_int_quiet
from .config import option_find_float
from .config import option_find_float_quiet
from .config import option_find_str
from .config import option_find_str_quiet
from .config import read_config

from .enum import *

import sys


class ListOf(object):
    def __init__(self, n, cls=None):
        self.__data = [None] * n
        if cls is not None:
            self.__data = [cls() for _ in self.__data]
        
    def __getitem__(self, item):
        return self.__data[item]

    def __setitem__(self, key, value):
        self.__data[key] = value
    
    @property
    def ref(self):
        return self.__data[0]
    
    @ref.setter
    def ref(self, value):
        self.__data[0] = value
        
    def __len__(self):
        return self.__data.__len__()

    def __iter__(self):
        return self.__data.__iter__()

    def __str__(self):
        return self.__data.__str__()

    def __repr__(self):
        return self.__data.__repr__()

    
def calloc(n, cls):
    """
    :param n: 
    :param cls: 
    :return: ListOf
    """
    return ListOf(n, cls)


class Layer(object):
    def __init__(self):
        super(Layer, self).__setattr__("__attr", {})
        self.type_string = "unknown"
        self.out_c = -1
        self.out_h = -1
        self.out_w = -1

    def __setattr__(self, key, value):
        if key == "__attr" and not isinstance(value, dict):
            raise Exception("__attr must be dict")
        __attr = super(Layer, self).__getattribute__("__attr")
        __attr[key] = value

    def __getattr__(self, item):
        __attr = super(Layer, self).__getattribute__("__attr")
        if item == "__attr":
            return __attr
        if item not in __attr:
            raise AttributeError(item)
        return __attr[item]

    def has(self, item):
        __attr = super(Layer, self).__getattribute__("__attr")
        return item in __attr and __attr[item] is not None

    def __str__(self):
        return "Layer[{}]({}, {}, {}, {})".format(self.type_string, -1, self.out_c, self.out_h, self.out_w)

    def __repr__(self):
        return "Layer[{}]({}, {}, {}, {})".format(self.type_string, -1, self.out_c, self.out_h, self.out_w)


def get_policy(s):
    if s == "random": return RANDOM
    if s == "poly": return POLY
    if s == "constant": return CONSTANT
    if s == "step": return STEP
    if s == "exp": return EXP
    if s == "sigmoid": return SIG
    if s == "steps": return STEPS
    if s == "random": return RANDOM
    return CONSTANT
        
        
class Network(object):
    def __init__(self, n, options):
        # type: (int, Session) -> None
        """
        :param n: layer number
        :param options: network options
        """
        '''
        make_network
        '''
        self.n = n
        self.layers = calloc(self.n, None)
        self.seen = calloc(1, int)
        self.t = calloc(1, int)
        self.cost = calloc(1, float)

        '''
        parse_net_options
        '''
        self.batch = option_find_int(options, "batch",1)
        self.learning_rate = option_find_float(options, "learning_rate", .001)
        self.momentum = option_find_float(options, "momentum", .9)
        self.decay = option_find_float(options, "decay", .0001)
        subdivs = option_find_int(options, "subdivisions", 1)
        self.time_steps = option_find_int_quiet(options, "time_steps", 1)
        self.notruth = option_find_int_quiet(options, "notruth", 0)
        self.batch //= subdivs
        self.batch *= self.time_steps
        self.subdivisions = subdivs
        self.random = option_find_int_quiet(options, "random", 0)

        self.adam = option_find_int_quiet(options, "adam", 0)
        if self.adam:
            self.B1 = option_find_float(options, "B1", .9)
            self.B2 = option_find_float(options, "B2", .999)
            self.eps = option_find_float(options, "eps", .0000001)

        self.h = option_find_int_quiet(options, "height",0)
        self.w = option_find_int_quiet(options, "width",0)
        self.c = option_find_int_quiet(options, "channels",0)
        self.inputs = option_find_int_quiet(options, "inputs", self.h * self.w * self.c)
        self.max_crop = option_find_int_quiet(options, "max_crop",self.w * 2)
        self.min_crop = option_find_int_quiet(options, "min_crop",self.w)
        self.max_ratio = option_find_float_quiet(options, "max_ratio", float(self.max_crop) / self.w)
        self.min_ratio = option_find_float_quiet(options, "min_ratio", float(self.min_crop) / self.w)
        self.center = option_find_int_quiet(options, "center",0)
        self.clip = option_find_float_quiet(options, "clip", 0)

        self.angle = option_find_float_quiet(options, "angle", 0)
        self.aspect = option_find_float_quiet(options, "aspect", 1)
        self.saturation = option_find_float_quiet(options, "saturation", 1)
        self.exposure = option_find_float_quiet(options, "exposure", 1)
        self.hue = option_find_float_quiet(options, "hue", 0)

        if self.inputs == 0 and self.h * self.w * self.c == 0:
            raise Exception("No input parameters supplied")

        policy_s = option_find_str(options, "policy", "constant")
        self.policy = get_policy(policy_s)
        self.burn_in = option_find_int_quiet(options, "burn_in", 0)
        self.power = option_find_float_quiet(options, "power", 4)
        if self.policy == STEP:
            self.step = option_find_int(options, "step", 1)
            self.scale = option_find_float(options, "scale", 1)
        elif self.policy == STEPS:
            self.scales = options.get_float_list("scales")
            self.steps = options.get_int_list("steps")
            if len(self.scales) == 0 or len(self.steps) == 0:
                raise Exception("STEPS policy must have steps and scales in cfg file")
            self.num_steps = len(self.steps)
        elif self.policy == EXP:
            self.gamma = option_find_float(options, "gamma", 1)
        elif self.policy == SIG:
            self.gamma = option_find_float(options, "gamma", 1)
            self.step = option_find_int(options, "step", 1)
        elif self.policy == POLY or self.policy == RANDOM:
            pass

        self.max_batches = option_find_int(options, "max_batches", 0)

        '''
        init outer variable
        '''
        self.outputs = None
        self.truths = None
        self.output = None
        self.input = None
        self.truth = None

        self.index = 0


def is_network(options):
    # type: (Session) -> bool
    return options.name == 'net' or options.name == 'network'


def make_network(n, options):
    # type: (int, Session) -> Network
    """
    :param n: layer number
    :param options: network options
    """
    return Network(n, options)


class SizeParams(object):
    def __init__(self, net):
        # type: (Network) -> None
        """
        :param net:
        """
        self.batch = None
        self.inputs = None
        self.h = None
        self.w = None
        self.c = None
        self.index = None
        self.time_steps = None
        self.net = None
        if net is not None:
            self.h = net.h
            self.w = net.w
            self.c = net.c
            self.inputs = net.inputs
            self.batch = net.batch
            self.time_steps = net.time_steps
            self.net = net
            self.index = 0


def fprintf(stream, format, *args):
    stream.write(format % args)


def get_network_output_layer(net):
    # type: (Network) -> Layer
    found_i = None
    for i in reversed(range(net.n)):
        if net.layers[i].type != COST:
            found_i = i
            break
    return net.layers[found_i]


def set_batch_network(net, b):
    # type: (Network, int) -> None
    net.batch = b
    for layer in net.layers:
        layer.batch = b
