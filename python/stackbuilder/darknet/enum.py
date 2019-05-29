#!python
# coding: UTF-8
"""
author: kier
"""

__enum_count = 0


def enum():
    global __enum_count
    i = __enum_count
    __enum_count += 1
    return i


def reset_enum(value=0):
    global __enum_count
    __enum_count = value


def strcmp(lhs, rhs, beg=1, end=-1):
    # type: (str, str, Union[int, None], Union[int, None]) -> bool
    return lhs == rhs[beg:end]


# learning_rate_policy
reset_enum()
CONSTANT = enum()
STEP = enum()
EXP = enum()
POLY = enum()
STEPS = enum()
SIG = enum()
RANDOM = enum()

# LAYER_TYPE
reset_enum()
CONVOLUTIONAL = enum()
DECONVOLUTIONAL = enum()
CONNECTED = enum()
MAXPOOL = enum()
SOFTMAX = enum()
DETECTION = enum()
DROPOUT = enum()
CROP = enum()
ROUTE = enum()
COST = enum()
NORMALIZATION = enum()
AVGPOOL = enum()
LOCAL = enum()
SHORTCUT = enum()
ACTIVE = enum()
RNN = enum()
GRU = enum()
LSTM = enum()
CRNN = enum()
BATCHNORM = enum()
NETWORK = enum()
XNOR = enum()
REGION = enum()
YOLO = enum()
ISEG = enum()
REORG = enum()
UPSAMPLE = enum()
LOGXENT = enum()
L2NORM = enum()
BLANK = enum()

# ACTIVATION
reset_enum()
LOGISTIC = enum()
RELU = enum()
RELIE = enum()
LINEAR = enum()
RAMP = enum()
TANH = enum()
PLSE = enum()
LEAKY = enum()
ELU = enum()
LOGGY = enum()
STAIR = enum()
HARDTAN = enum()
LHTAN = enum()
SELU = enum()

ACTIVATION_STRING = {
    0: "LOGISTIC",
    1: "RELU",
    2: "RELIE",
    3: "LINEAR",
    4: "RAMP",
    5: "TANH",
    6: "PLSE",
    7: "LEAKY",
    8: "ELU",
    9: "LOGGY",
    10: "STAIR",
    11: "HARDTAN",
    12: "LHTAN",
    13: "SELU",
}
