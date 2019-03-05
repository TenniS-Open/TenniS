#! python

"""
convert onnx_dtype 2 ts.dtype
"""

import tensorstack as ts
import onnx

"""
enum DataType {
    UNDEFINED = 0;
    // Basic types.
    FLOAT = 1;   // float
    UINT8 = 2;   // uint8_t
    INT8 = 3;    // int8_t
    UINT16 = 4;  // uint16_t
    INT16 = 5;   // int16_t
    INT32 = 6;   // int32_t
    INT64 = 7;   // int64_t
    STRING = 8;  // string
    BOOL = 9;    // bool

    // IEEE754 half-precision floating-point format (16 bits wide).
    // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10;

    DOUBLE = 11;
    UINT32 = 12;
    UINT64 = 13;
    COMPLEX64 = 14;     // complex with float32 real and imaginary components
    COMPLEX128 = 15;    // complex with float64 real and imaginary components

    // Non-IEEE floating-point format based on IEEE754 single-precision
    // floating-point number truncated to 16 bits.
    // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16;

    // Future extensions go here.
  }
"""

convert_pairs = [
    (onnx.TensorProto.FLOAT, ts.dtype.FLOAT32),
    (onnx.TensorProto.UINT8, ts.dtype.UINT8),
    (onnx.TensorProto.INT8, ts.dtype.INT8),
    (onnx.TensorProto.UINT16, ts.dtype.UINT16),
    (onnx.TensorProto.INT16, ts.dtype.INT16),
    (onnx.TensorProto.INT32, ts.dtype.INT32),
    (onnx.TensorProto.INT64, ts.dtype.INT64),
    (onnx.TensorProto.STRING, ts.dtype.CHAR8),
    (onnx.TensorProto.BOOL, ts.dtype.BOOLEAN),

    (onnx.TensorProto.FLOAT16, ts.dtype.FLOAT16),

    (onnx.TensorProto.DOUBLE, ts.dtype.FLOAT64),
    (onnx.TensorProto.UINT32, ts.dtype.UINT32),
    (onnx.TensorProto.UINT64, ts.dtype.UINT64),
    (onnx.TensorProto.COMPLEX64, ts.dtype.COMPLEX64),
    (onnx.TensorProto.COMPLEX128, ts.dtype.COMPLEX128),

    # (onnx.TensorProto.BFLOAT16, ts.dtype.BFLOAT16),
]

__onnx_dtype_to_dtype = { pair[0]: pair[1] for pair in convert_pairs }
__dtype_to_onnx_dtype = { pair[1]: pair[0] for pair in convert_pairs }


def to_onnx(dtype):
    # type: (int) -> onnx.TensorProto.DataType
    """
    convert ts.dtype to onnx.TensorProto.DataType
    :param dtype: int value of ts.dtype
    :return: onnx.TensorProto.DataType
    """
    if dtype not in __dtype_to_onnx_dtype:
        raise Exception("Not supported converting tensorstack dtype ={}".format(dtype))
    return __dtype_to_onnx_dtype[dtype]


def from_onnx(dtype):
    # type: (onnx.TensorProto.DataType) -> int
    """
    convert onnx.TensorProto.DataType to ts.dtype
    :param dtype: int value of onnx.TensorProto.DataType
    :return: int
    """
    if dtype not in __onnx_dtype_to_dtype:
        raise Exception("Not supported converting onnx dtype ={}".format(dtype))
    return __onnx_dtype_to_dtype[dtype]
