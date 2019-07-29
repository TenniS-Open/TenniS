#!/usr/bin/env python

"""
:author Kier
"""

import ctypes
from ctypes import *
from ctypes.util import find_library


TensorStackName = "TensorStack"
libTensorStack = find_library(TensorStackName)

lib = None
if libTensorStack is not None:
    try:
        lib = CDLL(libTensorStack, RTLD_GLOBAL)
    except:
        raise

msg = None
if lib is None:
    from . import libfinder
    lib, msg = libfinder.load_library(TensorStackName)


if lib is None:
    if msg is None:
        raise ImportError("Can find library: {}".format(TensorStackName))
    else:
        raise ImportError(msg)


def __TS_IMPORT(lib, sym, restype=None, *argtypes):
    # type: (CDLL, str, object, object) -> callable
    if not hasattr(lib, sym):
        import sys
        restype_str = "void" if restype is None else restype.__name__
        argtypes_str = ["void" if t is None else t.__name__ for t in argtypes]
        argtypes_str = ", ".join(argtypes_str)
        sys.stderr.write("[Warning]: Can not open sym: {} {}({})\n".format(restype_str, sym, argtypes_str))

        def log_error(*args, **kwargs):
            raise Exception("Use unloaded function: {}".format(sym))
        return log_error
    f = getattr(lib, sym)
    f.argtypes = argtypes
    f.restype = restype
    return f


def ts_api_check_pointer(pointer):
    # type: (ctypes._Pointer) -> None
    if not pointer:
        import sys
        message = ts_last_error_message()
        message = message.decode()
        # sys.stderr.write("[ERROR]: {}\n".format(message))
        raise Exception(message)


def ts_api_check_bool(value):
    # type: (ctypes.c_int) -> None
    if not value:
        import sys
        message = ts_last_error_message()
        message = message.decode()
        # sys.stderr.write("[ERROR]: {}\n".format(message))
        raise Exception(message)


""" ================================================================================================================ +++
common.h
"""
ts_last_error_message = __TS_IMPORT(lib, "ts_last_error_message", c_char_p)
ts_set_error_message = __TS_IMPORT(lib, "ts_set_error_message", None, c_char_p)

""" ================================================================================================================ +++
setup.h
"""
ts_setup = __TS_IMPORT(lib, "ts_setup", None)

""" ================================================================================================================ +++
stream.h
"""
ts_stream_write = CFUNCTYPE(c_ulonglong, c_void_p, c_void_p, c_ulonglong)

ts_stream_read = CFUNCTYPE(c_ulonglong, c_void_p, c_void_p, c_ulonglong)

""" ================================================================================================================ +++
module.h
"""
ts_SerializationFormat = c_int32
TS_BINARY = 0
TS_TEXT = 1


class ts_Module(Structure):
    pass


ts_Module_Load = __TS_IMPORT(lib, "ts_Module_Load", POINTER(ts_Module), c_char_p, ts_SerializationFormat)

ts_Module_LoadFromStream = __TS_IMPORT(lib, "ts_Module_LoadFromStream",
                                       POINTER(ts_Module), c_void_p, ts_stream_read, ts_SerializationFormat)

ts_free_Module = __TS_IMPORT(lib, "ts_free_Module", None, POINTER(ts_Module))

""" ================================================================================================================ +++
device.h
"""


class ts_Device(Structure):
    _fields_ = [
        ("type", c_char_p),
        ("id", c_int32),
    ]


""" ================================================================================================================ +++
tensor.h
"""
ts_DTYPE = c_int32
TS_VOID        = 0
TS_INT8        = 1
TS_UINT8       = 2
TS_INT16       = 3
TS_UINT16      = 4
TS_INT32       = 5
TS_UINT32      = 6
TS_INT64       = 7
TS_UINT64      = 8
TS_FLOAT32     = 10
TS_FLOAT64     = 11
TS_CHAR8       = 13


class ts_Tensor(Structure):
    pass


ts_InFlow = c_int32
TS_HOST = 0
TS_DEVICE = 1

ts_bool = c_int32

ts_new_Tensor = __TS_IMPORT(lib, "ts_new_Tensor", POINTER(ts_Tensor), POINTER(c_int32), c_int32, ts_DTYPE, c_void_p)

ts_free_Tensor = __TS_IMPORT(lib, "ts_free_Tensor", None, POINTER(ts_Tensor))

ts_Tensor_shape = __TS_IMPORT(lib, "ts_Tensor_shape", POINTER(c_int32), POINTER(ts_Tensor))

ts_Tensor_shape_size = __TS_IMPORT(lib, "ts_Tensor_shape_size", c_int32, POINTER(ts_Tensor))

ts_Tensor_dtype = __TS_IMPORT(lib, "ts_Tensor_dtype", ts_DTYPE, POINTER(ts_Tensor))

ts_Tensor_data = __TS_IMPORT(lib, "ts_Tensor_data", c_void_p, POINTER(ts_Tensor))

ts_Tensor_clone = __TS_IMPORT(lib, "ts_Tensor_clone", POINTER(ts_Tensor), POINTER(ts_Tensor))

ts_Tensor_sync_cpu = __TS_IMPORT(lib, "ts_Tensor_sync_cpu", ts_bool, POINTER(ts_Tensor))

ts_Tensor_cast = __TS_IMPORT(lib, "ts_Tensor_cast", POINTER(ts_Tensor), POINTER(ts_Tensor), ts_DTYPE)

ts_Tensor_reshape = __TS_IMPORT(lib, "ts_Tensor_reshape",
                                POINTER(ts_Tensor), POINTER(ts_Tensor), POINTER(c_int32), c_int32)

ts_new_Tensor_in_flow = __TS_IMPORT(lib, "ts_new_Tensor_in_flow",
                                    POINTER(ts_Tensor), ts_InFlow, POINTER(c_int32), c_int32, ts_DTYPE, c_void_p)

ts_Tensor_view_in_flow = __TS_IMPORT(lib, "ts_Tensor_view_in_flow", POINTER(ts_Tensor), POINTER(ts_Tensor), ts_InFlow)

ts_Tensor_field = __TS_IMPORT(lib, "ts_Tensor_field", POINTER(ts_Tensor), POINTER(ts_Tensor), c_int32)

ts_Tensor_packed = __TS_IMPORT(lib, "ts_Tensor_packed", ts_bool, POINTER(ts_Tensor))

ts_Tensor_fields_count = __TS_IMPORT(lib, "ts_Tensor_fields_count", c_int32, POINTER(ts_Tensor))

ts_Tensor_pack = __TS_IMPORT(lib, "ts_Tensor_pack", POINTER(ts_Tensor), POINTER(POINTER(ts_Tensor)), c_int32)

ts_Tensor_slice = __TS_IMPORT(lib, "ts_Tensor_slice", POINTER(ts_Tensor), POINTER(ts_Tensor), c_int32)

ts_Tensor_slice_v2 = __TS_IMPORT(lib, "ts_Tensor_slice_v2", POINTER(ts_Tensor), POINTER(ts_Tensor), c_int32, c_int32)

ts_Tensor_save = __TS_IMPORT(lib, "ts_Tensor_save", ts_bool, c_char_p, POINTER(ts_Tensor))

ts_Tensor_load = __TS_IMPORT(lib, "ts_Tensor_load", POINTER(ts_Tensor), c_char_p)


""" ================================================================================================================ +++
program.h
"""


class ts_Program(Structure):
    pass


ts_Program_Compile = __TS_IMPORT(lib, "ts_Program_Compile", POINTER(ts_Program), POINTER(ts_Module), POINTER(ts_Device))

ts_Program_Compile_v2 = __TS_IMPORT(lib, "ts_Program_Compile_v2",
                                    POINTER(ts_Program), POINTER(ts_Module), POINTER(ts_Device), c_char_p)

ts_free_Program = __TS_IMPORT(lib, "ts_free_Program", None, POINTER(ts_Program))

ts_Program_clone = __TS_IMPORT(lib, "ts_Program_clone", POINTER(ts_Program), POINTER(ts_Program))

ts_Program_input_count = __TS_IMPORT(lib, "ts_Program_input_count", c_int32, POINTER(ts_Program))

ts_Program_output_count = __TS_IMPORT(lib, "ts_Program_output_count", c_int32, POINTER(ts_Program))

ts_Program_set_operator_param = __TS_IMPORT(lib, "ts_Program_set_operator_param",
                                            c_int32, POINTER(ts_Program), c_char_p, c_char_p, POINTER(ts_Tensor))


""" ================================================================================================================ +++
image_filter.h
"""


class ts_ImageFilter(Structure):
    pass


ts_ResizeMethod = c_int32
TS_RESIZE_BILINEAR = 0
TS_RESIZE_BICUBIC = 1
TS_RESIZE_NEAREST = 2


ts_new_ImageFilter = __TS_IMPORT(lib, "ts_new_ImageFilter", POINTER(ts_ImageFilter), POINTER(ts_Device))

ts_free_ImageFilter = __TS_IMPORT(lib, "ts_free_ImageFilter", ts_bool, POINTER(ts_ImageFilter))

ts_ImageFilter_clear = __TS_IMPORT(lib, "ts_ImageFilter_clear", ts_bool, POINTER(ts_ImageFilter))

ts_ImageFilter_compile = __TS_IMPORT(lib, "ts_ImageFilter_compile", ts_bool, POINTER(ts_ImageFilter))

ts_ImageFilter_to_float = __TS_IMPORT(lib, "ts_ImageFilter_to_float", ts_bool, POINTER(ts_ImageFilter))

ts_ImageFilter_scale = __TS_IMPORT(lib, "ts_ImageFilter_scale", ts_bool, POINTER(ts_ImageFilter), c_float)

ts_ImageFilter_sub_mean = __TS_IMPORT(lib, "ts_ImageFilter_sub_mean",
                                      ts_bool, POINTER(ts_ImageFilter), POINTER(c_float), c_int32)

ts_ImageFilter_div_std = __TS_IMPORT(lib, "ts_ImageFilter_div_std",
                                     ts_bool, POINTER(ts_ImageFilter), POINTER(c_float), c_int32)

ts_ImageFilter_resize = __TS_IMPORT(lib, "ts_ImageFilter_resize",
                                    ts_bool, POINTER(ts_ImageFilter), c_int32, c_int32)

ts_ImageFilter_resize_scalar = __TS_IMPORT(lib, "ts_ImageFilter_resize_scalar",
                                           ts_bool, POINTER(ts_ImageFilter), c_int32)

ts_ImageFilter_center_crop = __TS_IMPORT(lib, "ts_ImageFilter_center_crop",
                                         ts_bool, POINTER(ts_ImageFilter), c_int32, c_int32)

ts_ImageFilter_channel_swap = __TS_IMPORT(lib, "ts_ImageFilter_channel_swap",
                                          ts_bool, POINTER(ts_ImageFilter), POINTER(c_int32), c_int32)

ts_ImageFilter_to_chw = __TS_IMPORT(lib, "ts_ImageFilter_to_chw", ts_bool, POINTER(ts_ImageFilter))

ts_ImageFilter_prewhiten = __TS_IMPORT(lib, "ts_ImageFilter_prewhiten", ts_bool, POINTER(ts_ImageFilter))

ts_ImageFilter_letterbox = __TS_IMPORT(lib, "ts_ImageFilter_letterbox",
                                       ts_bool, POINTER(ts_ImageFilter), c_int32, c_int32, c_float)

ts_ImageFilter_divided = __TS_IMPORT(lib, "ts_ImageFilter_divided",
                                     ts_bool, POINTER(ts_ImageFilter), c_int32, c_int32, c_float)

ts_ImageFilter_run = __TS_IMPORT(lib, "ts_ImageFilter_run",
                                 POINTER(ts_Tensor), POINTER(ts_ImageFilter), POINTER(ts_Tensor))

ts_ImageFilter_letterbox_v2 = __TS_IMPORT(lib, "ts_ImageFilter_letterbox_v2",
                                          ts_bool, POINTER(ts_ImageFilter), c_int32, c_int32, c_float, ts_ResizeMethod)

ts_ImageFilter_resize_v2 = __TS_IMPORT(lib, "ts_ImageFilter_resize_v2",
                                       ts_bool, POINTER(ts_ImageFilter), c_int32, c_int32, ts_ResizeMethod)

ts_ImageFilter_resize_scalar_v2 = __TS_IMPORT(lib, "ts_ImageFilter_resize_scalar_v2",
                                              ts_bool, POINTER(ts_ImageFilter), c_int32, ts_ResizeMethod)

ts_ImageFilter_force_color = __TS_IMPORT(lib, "ts_ImageFilter_force_color", ts_bool, POINTER(ts_ImageFilter))

ts_ImageFilter_force_gray = __TS_IMPORT(lib, "ts_ImageFilter_force_gray", ts_bool, POINTER(ts_ImageFilter))

ts_ImageFilter_force_gray_v2 = __TS_IMPORT(lib, "ts_ImageFilter_force_gray_v2",
                                           ts_bool, POINTER(ts_ImageFilter), POINTER(c_float), c_int32)

ts_ImageFilter_norm_image = __TS_IMPORT(lib, "ts_ImageFilter_norm_image", ts_bool, POINTER(ts_ImageFilter), c_float)


""" ================================================================================================================ +++
workbench.h
"""


class ts_Workbench(Structure):
    pass


ts_Workbench_Load = __TS_IMPORT(lib, "ts_Workbench_Load", POINTER(ts_Workbench), POINTER(ts_Module), POINTER(ts_Device))

ts_free_Workbench = __TS_IMPORT(lib, "ts_free_Workbench", None, POINTER(ts_Workbench))

ts_Workbench_clone = __TS_IMPORT(lib, "ts_Workbench_clone", POINTER(ts_Workbench), POINTER(ts_Workbench))

ts_Workbench_input = __TS_IMPORT(lib, "ts_Workbench_input", ts_bool, POINTER(ts_Workbench), c_int32, POINTER(ts_Tensor))

ts_Workbench_input_by_name = __TS_IMPORT(lib, "ts_Workbench_input_by_name",
                                         ts_bool, POINTER(ts_Workbench), c_char_p, POINTER(ts_Tensor))

ts_Workbench_run = __TS_IMPORT(lib, "ts_Workbench_run", ts_bool, POINTER(ts_Workbench))

ts_Workbench_output = __TS_IMPORT(lib, "ts_Workbench_output",
                                  ts_bool, POINTER(ts_Workbench), c_int32, POINTER(ts_Tensor))

ts_Workbench_output_by_name = __TS_IMPORT(lib, "ts_Workbench_output_by_name",
                                          ts_bool, POINTER(ts_Workbench), c_char_p, POINTER(ts_Tensor))

ts_Workbench_set_computing_thread_number = __TS_IMPORT(lib, "ts_Workbench_set_computing_thread_number",
                                                       ts_bool, POINTER(ts_Workbench), c_int32)

ts_Workbench_bind_filter = __TS_IMPORT(lib, "ts_Workbench_bind_filter",
                                       ts_bool, POINTER(ts_Workbench), c_int32, POINTER(ts_ImageFilter))

ts_Workbench_bind_filter_by_name = __TS_IMPORT(lib, "ts_Workbench_bind_filter_by_name",
                                               ts_bool, POINTER(ts_Workbench), c_char_p, POINTER(ts_ImageFilter))

ts_new_Workbench = __TS_IMPORT(lib, "ts_new_Workbench", POINTER(ts_Workbench), POINTER(ts_Device))

ts_Workbench_setup = __TS_IMPORT(lib, "ts_Workbench_setup", ts_bool, POINTER(ts_Workbench), POINTER(ts_Program))

ts_Workbench_setup_context = __TS_IMPORT(lib, "ts_Workbench_setup_context", ts_bool, POINTER(ts_Workbench))

ts_Workbench_compile = __TS_IMPORT(lib, "ts_Workbench_compile",
                                   POINTER(ts_Program), POINTER(ts_Workbench), POINTER(ts_Module))

ts_Workbench_setup_device = __TS_IMPORT(lib, "ts_Workbench_setup_device", ts_bool, POINTER(ts_Workbench))

ts_Workbench_setup_runtime = __TS_IMPORT(lib, "ts_Workbench_setup_runtime", ts_bool, POINTER(ts_Workbench))

ts_Workbench_input_count = __TS_IMPORT(lib, "ts_Workbench_input_count", c_int32, POINTER(ts_Workbench))

ts_Workbench_output_count = __TS_IMPORT(lib, "ts_Workbench_output_count", c_int32, POINTER(ts_Workbench))

ts_Workbench_run_hook = __TS_IMPORT(lib, "ts_Workbench_run_hook",
                                    ts_bool, POINTER(ts_Workbench), POINTER(c_char_p), c_int32)

ts_Workbench_Load_v2 = __TS_IMPORT(lib, "ts_Workbench_Load_v2",
                                   POINTER(ts_Workbench), POINTER(ts_Module), POINTER(ts_Device), c_char_p)

ts_Workbench_compile_v2 = __TS_IMPORT(lib, "ts_Workbench_compile_v2",
                                      POINTER(ts_Program), POINTER(ts_Workbench), POINTER(ts_Module), c_char_p)

ts_Workbench_set_operator_param = __TS_IMPORT(lib, "ts_Workbench_set_operator_param",
                                              c_int32, POINTER(ts_Workbench), c_char_p, c_char_p, POINTER(ts_Tensor))

ts_Workbench_summary = __TS_IMPORT(lib, "ts_Workbench_summary",
                                   c_char_p, POINTER(ts_Workbench))


""" ================================================================================================================ +++
intime.h
"""

ts_intime_transpose = __TS_IMPORT(
    lib, "ts_intime_transpose", POINTER(ts_Tensor),
    POINTER(ts_Tensor),     # x
    POINTER(c_int32),       # shuffle
    c_int32,                # len
)

ts_intime_sigmoid = __TS_IMPORT(
    lib, "ts_intime_sigmoid", POINTER(ts_Tensor),
    POINTER(ts_Tensor),     # x
)

ts_intime_gather = __TS_IMPORT(
    lib, "ts_intime_gather", POINTER(ts_Tensor),
    POINTER(ts_Tensor),     # x
    POINTER(ts_Tensor),     # indices
    c_int32,                # len
)

ts_intime_concat = __TS_IMPORT(
    lib, "ts_intime_concat", POINTER(ts_Tensor),
    POINTER(POINTER(ts_Tensor)),    # x
    c_int32,                        # len
    c_int32,                        # dim
)

ts_intime_softmax = __TS_IMPORT(
    lib, "ts_intime_softmax", POINTER(ts_Tensor),
    POINTER(ts_Tensor),     # x
    c_int32,                # dim
    ts_bool,                # smooth
)

ts_intime_pad = __TS_IMPORT(
    lib, "ts_intime_pad", POINTER(ts_Tensor),
    POINTER(ts_Tensor),     # x
    POINTER(ts_Tensor),     # padding
    c_float,                # padding_value
)

ts_intime_cast = __TS_IMPORT(
    lib, "ts_intime_cast", POINTER(ts_Tensor),
    POINTER(ts_Tensor),     # x
    ts_DTYPE,               # dtype
)

ts_intime_resize2d = __TS_IMPORT(
    lib, "ts_intime_resize2d", POINTER(ts_Tensor),
    POINTER(ts_Tensor),     # x
    POINTER(ts_Tensor),     # size
    c_int32,                # method
)

ts_intime_affine_sample2d = __TS_IMPORT(
    lib, "ts_intime_affine_sample2d", POINTER(ts_Tensor),
    POINTER(ts_Tensor),     # x
    POINTER(ts_Tensor),     # size
    POINTER(ts_Tensor),     # affine
    c_int32,                # dim
    c_float,                # outer_value
    c_int32,                # method
)


""" ================================================================================================================ +++
operator.h
"""


class ts_OperatorParams(Structure):
    pass


class ts_OperatorContext(Structure):
    pass


ts_OperatorParams_get = __TS_IMPORT(lib, "ts_OperatorParams_get",
                                    POINTER(ts_Tensor), POINTER(ts_OperatorParams), c_char_p)

ts_new_Operator = CFUNCTYPE(c_void_p)

ts_free_Operator = CFUNCTYPE(None, c_void_p)

ts_Operator_init = CFUNCTYPE(None, c_void_p, POINTER(ts_OperatorParams), POINTER(ts_OperatorContext))

ts_Operator_init_ex = CFUNCTYPE(ts_bool, c_void_p, POINTER(ts_OperatorParams), POINTER(ts_OperatorContext))

ts_Operator_infer = CFUNCTYPE(c_void_p,
                              c_void_p, c_int32, POINTER(POINTER(ts_Tensor)), POINTER(ts_OperatorContext))

ts_Operator_run = CFUNCTYPE(c_void_p,
                            c_void_p, c_int32, POINTER(POINTER(ts_Tensor)), POINTER(ts_OperatorContext))

ts_Operator_Register = __TS_IMPORT(lib, "ts_Operator_Register", None,
                                   c_char_p,
                                   c_char_p,
                                   ts_new_Operator,
                                   ts_free_Operator,
                                   ts_Operator_init,
                                   ts_Operator_infer,
                                   ts_Operator_run)

ts_Operator_RegisterEx = __TS_IMPORT(lib, "ts_Operator_RegisterEx", None,
                                     c_char_p,
                                     c_char_p,
                                     ts_new_Operator,
                                     ts_free_Operator,
                                     ts_Operator_init_ex,
                                     ts_Operator_infer,
                                     ts_Operator_run)

ts_Operator_Throw = __TS_IMPORT(lib, "ts_Operator_Throw", None, c_char_p)

ts_Operator_ThrowV2 = __TS_IMPORT(lib, "ts_Operator_ThrowV2", None, c_char_p, c_char_p, c_int32)


def TS_C_THROW(message):
    """
    Notice this function only can be called in C codes.
    :param message:
    :return:
    """
    try:
        raise Exception
    except:
        import sys
        import os
        f = sys.exc_info()[2].tb_frame.f_back
        filename = os.path.normcase(f.f_code.co_filename)
        linenumber = f.f_lineno
        functionname = f.f_code.co_name
        # use python like message format, not using C style ts_Operator_ThrowV2
        format_message = "File \"{}\", line {}, in {}: raise Exception(\"{}\")\n"
        ts_Operator_Throw(format_message.format(filename, linenumber, functionname, message))

