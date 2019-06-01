from . import _C
from . import dtype as DC
import numpy

VOID         = _C.TS_VOID
INT8         = _C.TS_INT8
UINT8        = _C.TS_UINT8
INT16        = _C.TS_INT16
UINT16       = _C.TS_UINT16
INT32        = _C.TS_INT32
UINT32       = _C.TS_UINT32
INT64        = _C.TS_INT64
UINT64       = _C.TS_UINT64
FLOAT32      = _C.TS_FLOAT32
FLOAT64      = _C.TS_FLOAT64
CHAR8        = _C.TS_CHAR8


HOST = _C.TS_HOST,
DEVICE = _C.TS_DEVICE,


class _Shared(object):
    def __init__(self, ptr, deleter=None):
        self.__ptr = ptr
        self.__deleter = deleter

    def __del__(self):
        if self.__deleter is not None:
            self.__deleter(self.__ptr)

    @property
    def raw(self):
        return self.__ptr


class Tensor(object):
    def __init__(self, obj=None, dtype=None, shape=None, in_flow=None, borrow=False):
        """

        :param obj: Union[_C.ts_Tensor, numpy.ndarray, None]
        :param dtype:
        :param shape:
        :param in_flow:
        :param borrow: working in obj is _C.ts_Tensor,
        """
        """
        _C.ts_Tensor
        """
        if isinstance(obj, _C.ts_Tensor):
            self.__shared_tensor = _Shared(obj, None if borrow else _C.ts_free_Tensor)
            return

        """
        numpy.ndarray
        """
        if obj is not None:
            np = numpy.ascontiguousarray(obj, dtype=None if dtype is None else DC.to_numpy_dtype(dtype=dtype))
            dtype = np.dtype

            if shape is not None:
                np = numpy.reshape(np, newshape=shape)
            else:
                shape = np.shape

            np_shape = numpy.ascontiguousarray(shape, dtype=numpy.int32)

            c_len = len(np_shape)
            c_shape = np_shape.ctypes.data_as(_C.POINTER(_C.c_int32))
            c_dtype = DC.to_ts_dtype(dtype)
            c_data = np.ctypes.data_as(_C.c_void_p)

            c_tensor = None
            if in_flow is None:
                c_tensor = _C.ts_new_Tensor(c_shape, c_len, c_dtype, c_data)
            else:
                c_in_flow = in_flow
                c_tensor = _C.ts_new_Tensor_in_flow(c_in_flow, c_shape, c_len, c_dtype, c_data)
            _C.ts_api_check_pointer(c_tensor)
            self.__shared_tensor = _Shared(c_tensor, _C.ts_free_Tensor)
            return

        """
        new tensor
        """
        self.__c_tensor = None
        if dtype and shape:
            if dtype is None:
                dtype = _C.TS_FLOAT32
            else:
                dtype = DC.to_ts_dtype(dtype)
            if shape is None:
                shape = ()

            np_shape = numpy.ascontiguousarray(shape, dtype=numpy.int32)

            c_len = len(np_shape)
            c_shape = np_shape.ctypes.data_as(_C.POINTER(_C.c_int32))
            c_dtype = dtype

            c_tensor = None
            if in_flow is None:
                c_tensor = _C.ts_new_Tensor(c_shape, c_len, c_dtype, None)
            else:
                c_in_flow = in_flow
                c_tensor = _C.ts_new_Tensor_in_flow(c_in_flow, c_shape, c_len, c_dtype, None)
            _C.ts_api_check_pointer(c_tensor)
            self.__shared_tensor = _Shared(c_tensor, _C.ts_free_Tensor)

    @property
    def _as_parameter_(self):
        return self.__shared_tensor.raw

    @property
    def numpy(self):
        c_tensor = self.__shared_tensor.raw
        if c_tensor is None:
            return None

        c_flag = _C.ts_Tensor_sync_cpu(c_tensor)
        _C.ts_api_check_bool(c_flag)
        c_dtype = _C.ts_Tensor_dtype(c_tensor)
        c_data = _C.ts_Tensor_data(c_tensor)
        c_shape = _C.ts_Tensor_shape(c_tensor)
        c_shape_size = _C.ts_Tensor_shape_size(c_tensor)
        shape = [c_shape[i] for i in range(c_shape_size)]
        print c_dtype
        c_dtype_data = _C.cast(c_data, _C.POINTER(DC.to_ctypes(c_dtype)))
        print c_dtype_data
        np = numpy.ctypeslib.as_array(c_dtype_data, shape=shape).copy()
        return np

