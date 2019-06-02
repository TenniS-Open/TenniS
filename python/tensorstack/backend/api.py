from . import _C
from . import dtype as DC
import numpy
import sys


class _Shared(object):
    def __init__(self, ptr, deleter=None):
        self.__ptr = ptr
        self.__deleter = deleter

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.__ptr is not None and self.__deleter is not None:
            self.__deleter(self.__ptr)
            self.__ptr = None

    def release(self):
        ptr = self.__ptr
        self.__ptr = None
        return ptr

    @property
    def raw(self):
        return self.__ptr

    def get(self):
        return self.__ptr


def last_error_message():
    # type: () -> str
    """
    :return: Last error message
    """
    return str(_C.ts_last_error_message())


def setup():
    # type: () -> None
    _C.ts_setup()


BINARY = _C.TS_BINARY
TEXT = _C.TS_TEXT


def compatible_string(obj):
    # type: (object) -> object
    if sys.version > '3':
        pass
    else:
        if isinstance(obj, unicode):
            return str(obj)
    return obj


class Module(object):
    def __init__(self, obj=None, borrow=False):
        self.__shared = _Shared(None)
        """
        Module
        """
        if isinstance(obj, Module):
            self.__shared = obj.__shared
            return
        """
        _C.ts_Module
        """
        if isinstance(obj, _C.POINTER(_C.ts_Module)):
            self.__shared = _Shared(obj, None if borrow else _C.ts_free_Module)
            return

        if obj is not None:
            raise Exception("argument {}: expected Module or POINTER(ts_Module) instance instead of {}".
                            format(1, type(obj).__name__))

    def dispose(self):
        self.__shared.dispose()

    @property
    def _as_parameter_(self):
        return self.__shared.raw

    @property
    def raw(self):
        return self._as_parameter_

    @staticmethod
    def Load(module, format=BINARY):
        # type: (Union[str, file], int) -> Module
        assert format in {BINARY, TEXT}
        module = compatible_string(module)
        if isinstance(module, str):
            module = _C.ts_Module_Load(module, format)
        elif hasattr(module, "read"):
            obj = _C.py_object(module)
            pobj = _C.pointer(obj)

            def stream_read(obj, data, count):
                obj = _C.cast(obj, _C.POINTER(_C.py_object))
                stream = obj.contents.value
                cbytes = stream.read(count)
                read_size = len(cbytes)
                _C.memmove(data, cbytes, min(count, read_size))
                return read_size

            c_stream_read = _C.ts_stream_read(stream_read)
            module = _C.ts_Module_LoadFromStream(_C.cast(pobj, _C.c_void_p), c_stream_read, format)
        else:
            raise Exception("argument {}: expected str or file instance instead of {}".
                            format(1, type(module).__name__))
        _C.ts_api_check_pointer(module)
        return Module(module)


class Device(object):
    def __init__(self, type="cpu", id=0):
        # type: (str, int) -> None
        self.__object = _C.ts_Device(type, id)

    @property
    def _as_parameter_(self):
        # type: () -> _C.POINTER(_C.ts_Device)
        return _C.byref(self.__object)

    @property
    def raw(self):
        return self._as_parameter_

    @property
    def ref(self):
        # type: () -> _C.ts_Device
        return self.__object


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


class Tensor(object):
    def __init__(self, obj=None, dtype=None, shape=None, in_flow=None, borrow=False):
        """

        :param obj: Union[_C.ts_Tensor, numpy.ndarray, None]
        :param dtype:
        :param shape:
        :param in_flow:
        :param borrow: working in obj is _C.ts_Tensor,
        """
        self.__shared = _Shared(None)
        """
        Tensor
        """
        if isinstance(obj, Tensor):
            self.__shared = obj.__shared
            return
        """
        _C.ts_Tensor
        """
        if isinstance(obj, _C.POINTER(_C.ts_Tensor)):
            self.__shared = _Shared(obj, None if borrow else _C.ts_free_Tensor)
            return

        """
        numpy.ndarray or array object
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
                assert in_flow in {HOST, DEVICE}
                c_in_flow = in_flow
                c_tensor = _C.ts_new_Tensor_in_flow(c_in_flow, c_shape, c_len, c_dtype, c_data)
            _C.ts_api_check_pointer(c_tensor)
            self.__shared = _Shared(c_tensor, _C.ts_free_Tensor)
            return

        """
        new tensor
        """
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
            self.__shared = _Shared(c_tensor, _C.ts_free_Tensor)
            return

    def dispose(self):
        self.__shared.dispose()

    @property
    def _as_parameter_(self):
        return self.__shared.raw

    @property
    def raw(self):
        return self._as_parameter_

    def __nonzero__(self):
        return bool(self.__shared.raw)

    @property
    def numpy(self):
        c_tensor = self.__shared.raw
        if c_tensor is None:
            return None

        c_flag = _C.ts_Tensor_sync_cpu(c_tensor)
        _C.ts_api_check_bool(c_flag)
        c_dtype = _C.ts_Tensor_dtype(c_tensor)
        c_data = _C.ts_Tensor_data(c_tensor)
        c_shape = _C.ts_Tensor_shape(c_tensor)
        c_shape_size = _C.ts_Tensor_shape_size(c_tensor)
        shape = [c_shape[i] for i in range(c_shape_size)]
        c_dtype_data = _C.cast(c_data, _C.POINTER(DC.to_ctypes(c_dtype)))
        np = numpy.ctypeslib.as_array(c_dtype_data, shape=shape).copy()
        return np

    @property
    def shape(self):
        # type: () -> tuple
        c_tensor = self.__shared.raw
        c_shape = _C.ts_Tensor_shape(c_tensor)
        if not c_shape:
            return tuple()
        c_shape_size = _C.ts_Tensor_shape_size(c_tensor)
        shape = [c_shape[i] for i in range(c_shape_size)]

        return tuple(shape)

    @property
    def dims(self):
        # type: () -> int
        return int(_C.ts_Tensor_shape_size(self))

    @property
    def shape_size(self):
        # type: () -> int
        return self.dims

    @property
    def dtype(self):
        # type: () -> int
        return int(_C.ts_Tensor_dtype(self))

    def copy(self):
        # type: () -> Tensor
        dolly = _C.ts_Tensor_clone(self)
        _C.ts_api_check_pointer(dolly)

        return Tensor(dolly)

    def clone(self):
        # type: () -> Tensor
        return self.copy()

    def sync_cpu(self):
        # type: () -> None
        _C.ts_api_check_bool(_C.ts_Tensor_sync_cpu())

    def cast(self, dtype):
        # type: (Uinon[int, type]) -> Tensor
        dtype = DC.to_ts_dtype(dtype)
        x = _C.ts_Tensor_cast(self, dtype)
        _C.ts_api_check_pointer(x)
        return Tensor(x)

    def reshape(self, shape):
        # type: (tuple) -> Tensor
        np_shape = numpy.ascontiguousarray(shape, dtype=numpy.int32)

        c_len = len(np_shape)
        c_shape = np_shape.ctypes.data_as(_C.POINTER(_C.c_int32))

        x = _C.ts_Tensor_reshape(self, c_shape, c_len)
        _C.ts_api_check_pointer(x)

        return Tensor(x)

    def view(self, in_flow):
        # type: (int) -> Tensor
        assert in_flow in {HOST, DEVICE}
        x =_C.ts_Tensor_view_in_flow(self, in_flow)
        _C.ts_api_check_pointer(x)
        return Tensor(x)

    def field(self, index):
        # type: (int) -> Tensor
        x = _C.ts_Tensor_field(self, index)
        _C.ts_api_check_pointer(x)
        return Tensor(x)

    def __getitem__(self, item):
        # type: (int) -> Tensor
        return self.field(item)

    def packed(self):
        # type: () -> bool
        return _C.ts_Tensor_packed(self)

    def fields_count(self):
        # type: () -> int
        return _C.ts_Tensor_fields_count(self)

    def __len__(self):
        # type: () -> int
        return self.fields_count()

    @staticmethod
    def Pack(fileds):
        # type: (List[Tensor]) -> Tensor
        if isinstance(fileds, Tensor):
            return fileds
        if isinstance(fileds, (tuple, list)):
            tensors = [Tensor(field) for field in fileds]
            c_tensors = [tensor.raw for tensor in tensors]
            c_len = len(tensors)
            c_fields = (_C.POINTER(_C.ts_Tensor) * c_len)(*c_tensors)
            x = _C.ts_Tensor_pack(c_fields, c_len)
            _C.ts_api_check_pointer(x)
            return Tensor(x)

        raise Exception("Fields must be list of Tensor")

    def __iter__(self):
        class Iteration(object):
            def __init__(self, tensor):
                # type: (Tensor) -> None
                self.__tensor = tensor
                self.__count = tensor.fields_count()
                self.__i = 0

            def next(self):
                if self.__i >= self.__count:
                    raise StopIteration
                x = self.__tensor[self.__i]
                self.__i += 1
                return x
        return Iteration(self)

    def unpack(self):
        # type () -> Tuple[Tensor]
        count = self.fields_count()
        if count == 1:
            return self,
        return (x for x in self)

    def slice(self, beg, end=None):
        # type: (int, int) -> Tensor
        x = None
        if end:
            x = _C.ts_Tensor_slice_v2(self, beg, end)
        else:
            x = _C.ts_Tensor_slice(self, beg)
        _C.ts_api_check_pointer(x)
        return Tensor(x)

    @staticmethod
    def Save(path, tensor):
        # type: (str, Tensor) -> None
        c_flag = _C.ts_Tensor_save(path, tensor.raw)
        _C.ts_api_check_bool(c_flag)

    @staticmethod
    def Load(path):
        # type: (str) -> Tensor
        x = _C.ts_Tensor_load(path)
        _C.ts_api_check_pointer(x)
        return Tensor(x)


