from . import _C
from . import dtype as DC
import numpy
import sys


def _to_ctypes_array(array, dtype):
    ctypes_dtype = DC.to_ctypes(dtype)
    numpy_dtype = DC.to_numpy_dtype(dtype)
    np_array = numpy.ascontiguousarray(array, dtype=numpy_dtype)
    c_len = len(np_array)
    c_array = np_array.ctypes.data_as(_C.POINTER(ctypes_dtype))
    return c_array, c_len, np_array


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
    message = _C.ts_last_error_message()
    return message.decode()


def set_error_message(message):
    # type: (str) -> None
    message = message.encode()
    _C.ts_set_error_message(message)


def setup():
    # type: () -> None
    _C.ts_setup()


BINARY = _C.TS_BINARY
TEXT = _C.TS_TEXT


def _compatible_string(obj):
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
        module = _compatible_string(module)
        if isinstance(module, str):
            module = module.encode()
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

    @staticmethod
    def Fusion(in_module, in_out_slot, out_module, out_in_slot):
        # type: (Module, int, Module, int) -> Workbench
        if not isinstance(in_module, (Module, _C.POINTER(_C.ts_Module))):
            raise Exception("argument {}: expected Module or POINTER(ts_Module) instance instead of {}".
                            format(1, type(in_module).__name__))
        if not isinstance(out_module, (Module, _C.POINTER(_C.ts_Module))):
            raise Exception("argument {}: expected Module or POINTER(ts_Module) instance instead of {}".
                            format(3, type(out_module).__name__))
        try:
            in_out_slot = int(in_out_slot)
        except:
            raise Exception("argument {}: expected int instead of {}".
                            format(2, type(in_out_slot).__name__))
        try:
            out_in_slot = int(out_in_slot)
        except:
            raise Exception("argument {}: expected int instead of {}".
                            format(4, type(out_in_slot).__name__))

        module = _C.ts_Module_Fusion(in_module, in_out_slot, out_module, out_in_slot)
        _C.ts_api_check_pointer(module)
        return Module(module)


class Device(object):
    def __init__(self, type="cpu", id=0):
        # type: (str, int) -> None
        type = type.encode()
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


class Tensor(object):
    class InFlow(object):
        HOST = _C.TS_HOST
        DEVICE = _C.TS_DEVICE

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
        string
        """
        if isinstance(obj, numpy.ndarray) and (obj.dtype.type == numpy.string_ or obj.dtype.type == numpy.str_):
            obj = str(obj)
        str_obj = _compatible_string(obj)
        if isinstance(str_obj, str):
            obj = str_obj
            obj = obj.encode()
            if dtype is not None and dtype != CHAR8:
                raise ValueError("dtype should be None or CHAR8 with type(obj)==str")
            if shape is not None and shape != (len(obj),):
                raise ValueError("shape should be None or [{}] with type(obj)==str".format(len(obj)))

            shape = (len(obj),)

            c_shape, c_len, _ = _to_ctypes_array(shape, _C.c_int32)
            c_dtype = CHAR8
            c_str = _C.c_char_p(obj)
            c_data = _C.cast(c_str, _C.c_void_p).value

            c_tensor = None
            if in_flow is None:
                c_tensor = _C.ts_new_Tensor(c_shape, c_len, c_dtype, c_data)
            else:
                assert in_flow in {self.InFlow.HOST, self.InFlow.DEVICE}
                c_in_flow = _C.c_int32(in_flow)
                c_tensor = _C.ts_new_Tensor_in_flow(c_in_flow, c_shape, c_len, c_dtype, c_data)
            _C.ts_api_check_pointer(c_tensor)
            self.__shared = _Shared(c_tensor, None if borrow else _C.ts_free_Tensor)
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

            c_shape, c_len, _ = _to_ctypes_array(shape, _C.c_int32)

            c_dtype = VOID
            c_data = None
            if dtype.type == numpy.object_:
                if np.shape == ():
                    pass
                else:
                    object_item = np.flatten()[0]
                    if object_item is not None:
                        raise NotImplementedError("array of type={}", type(object_item))
            else:
                c_dtype = DC.to_ts_dtype(dtype)
                c_data = np.ctypes.data_as(_C.c_void_p)

            c_tensor = None
            if in_flow is None:
                c_tensor = _C.ts_new_Tensor(c_shape, c_len, c_dtype, c_data)
            else:
                assert in_flow in {self.InFlow.HOST, self.InFlow.DEVICE}
                c_in_flow = _C.c_int32(in_flow)
                c_tensor = _C.ts_new_Tensor_in_flow(c_in_flow, c_shape, c_len, c_dtype, c_data)
            _C.ts_api_check_pointer(c_tensor)
            self.__shared = _Shared(c_tensor, None if borrow else _C.ts_free_Tensor)
            return

        """
        new tensor
        """
        if dtype is not None or shape is not None:
            if dtype is None:
                dtype = _C.TS_FLOAT32
            else:
                dtype = DC.to_ts_dtype(dtype)
            if shape is None:
                shape = ()

            c_shape, c_len, _ = _to_ctypes_array(shape, _C.c_int32)
            c_dtype = dtype

            c_tensor = None
            if in_flow is None:
                c_tensor = _C.ts_new_Tensor(c_shape, c_len, c_dtype, None)
            else:
                assert in_flow in {self.InFlow.HOST, self.InFlow.DEVICE}
                c_in_flow = _C.c_int32(in_flow)
                c_tensor = _C.ts_new_Tensor_in_flow(c_in_flow, c_shape, c_len, c_dtype, None)
            _C.ts_api_check_pointer(c_tensor)
            self.__shared = _Shared(c_tensor, None if borrow else _C.ts_free_Tensor)
            return

    def dispose(self):
        self.__shared.dispose()

    @property
    def _as_parameter_(self):
        return self.__shared.raw

    @property
    def raw(self):
        return self._as_parameter_

    def release(self):
        return self.__shared.release()

    def __nonzero__(self):
        return bool(self.__shared.raw)

    @property
    def numpy(self):
        c_tensor = self.__shared.raw
        if c_tensor is None:
            return None

        c_dtype = _C.ts_Tensor_dtype(c_tensor)

        if int(c_dtype) == CHAR8:
            return numpy.asarray(self.str)

        if int(c_dtype) == VOID:
            self_shape = self.shape
            if self_shape == ():
                return None
            return numpy.reshape([None] * numpy.prod(self_shape), self_shape)

        c_flag = _C.ts_Tensor_sync_cpu(c_tensor)
        _C.ts_api_check_bool(c_flag)
        c_data = _C.ts_Tensor_data(c_tensor)
        c_shape = _C.ts_Tensor_shape(c_tensor)
        c_shape_size = _C.ts_Tensor_shape_size(c_tensor)
        shape = [c_shape[i] for i in range(c_shape_size)]
        c_dtype_data = _C.cast(c_data, _C.POINTER(DC.to_ctypes(c_dtype)))
        np = numpy.ctypeslib.as_array(c_dtype_data, shape=tuple(shape)).copy()
        return np

    def __array__(self):
        return self.numpy

    @property
    def str(self):
        # type: () -> Union[str, None]
        c_tensor = self.__shared.raw
        if c_tensor is None:
            return None

        c_dtype = _C.ts_Tensor_dtype(c_tensor)

        if int(c_dtype) != CHAR8:
            raise NotImplementedError("Can not convert dtype {} to {}".format(c_dtype, CHAR8))

        c_flag = _C.ts_Tensor_sync_cpu(c_tensor)
        _C.ts_api_check_bool(c_flag)
        c_data = _C.ts_Tensor_data(c_tensor)
        c_shape = _C.ts_Tensor_shape(c_tensor)
        c_shape_size = _C.ts_Tensor_shape_size(c_tensor)
        shape = [c_shape[i] for i in range(c_shape_size)]

        count = numpy.prod(shape)

        c_char_buffer = _C.cast(c_data, _C.POINTER(_C.c_char))

        _C.resize(c_char_buffer, max(count + 1, 8))
        c_char_buffer[count] = '\0'.encode()

        c_str = _C.cast(c_char_buffer, _C.c_char_p)

        return str(c_str.value.decode())


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
        c_shape, c_len, _ = _to_ctypes_array(shape, _C.c_int32)
        x = _C.ts_Tensor_reshape(self, c_shape, c_len)
        _C.ts_api_check_pointer(x)

        return Tensor(x)

    def view(self, in_flow):
        # type: (int) -> Tensor
        assert in_flow in {self.InFlow.HOST, self.InFlow.DEVICE}
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
            fileds = (fileds,)
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
        if end is not None:
            beg = _C.c_int32(beg)
            end = _C.c_int32(end)
            x = _C.ts_Tensor_slice_v2(self, beg, end)
        else:
            beg = _C.c_int32(beg)
            x = _C.ts_Tensor_slice(self, beg)
        _C.ts_api_check_pointer(x)
        return Tensor(x)

    @staticmethod
    def Save(path, tensor):
        # type: (str, Tensor) -> None
        path = path.encode()
        tensor = Tensor(tensor)
        c_flag = _C.ts_Tensor_save(path, tensor.raw)
        _C.ts_api_check_bool(c_flag)

    @staticmethod
    def Load(path):
        # type: (str) -> Tensor
        path = path.encode()
        x = _C.ts_Tensor_load(path)
        _C.ts_api_check_pointer(x)
        return Tensor(x)


class Program(object):
    def __init__(self, obj=None, borrow=False):
        self.__shared = _Shared(None)
        """
        Program
        """
        if isinstance(obj, Program):
            self.__shared = obj.__shared
            return
        """
        _C.ts_Program
        """
        if isinstance(obj, _C.POINTER(_C.ts_Program)):
            self.__shared = _Shared(obj, None if borrow else _C.ts_free_Program)
            return

        if obj is not None:
            raise Exception("argument {}: expected Program or POINTER(ts_Program) instance instead of {}".
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
    def Compile(module, device, options=None):
        # type: (Module, Device, str) -> Program
        if not isinstance(module, (Module, _C.POINTER(_C.ts_Module))):
            raise Exception("argument {}: expected Module or POINTER(ts_Module) instance instead of {}".
                            format(1, type(module).__name__))
        if not isinstance(device, (Device, _C.POINTER(_C.ts_Device))):
            raise Exception("argument {}: expected Device or POINTER(ts_Device) instance instead of {}".
                            format(2, type(device).__name__))
        options = _compatible_string(options)
        if options is not None and not isinstance(options, str):
            raise Exception("argument {}: expected None or str instance instead of {}".
                            format(3, type(options).__name__))
        c_program = _C.POINTER(_C.ts_Program)
        if isinstance(options, str):
            options = options.encode()
            c_program = _C.ts_Program_Compile_v2(module, device, options)
        else:
            c_program = _C.ts_Program_Compile(module, device)
        _C.ts_api_check_pointer(c_program)
        return Program(c_program)

    def clone(self):
        # type: () -> Program
        x = _C.ts_Program_clone(self)
        _C.ts_api_check_pointer(x)
        return Program(x)

    def copy(self):
        # type: () -> Program
        return self.clone()

    def input_count(self):
        # type: () -> int
        return _C.ts_Program_input_count(self)

    def output_count(self):
        # type: () -> int
        return _C.ts_Program_output_count(self)

    def set_operator_param(self, node_name, param, value):
        # type: (str, str, Tensor) -> None
        node_name = node_name.encode()
        param = param.encode()
        value = Tensor(value)
        _C.ts_api_check_bool(_C.ts_Program_set_operator_param(self, node_name, param, value))


class ImageFilter(object):
    class ResizeMethod(object):
        BILINEAR = _C.TS_RESIZE_BILINEAR
        BICUBIC = _C.TS_RESIZE_BICUBIC
        NEAREST = _C.TS_RESIZE_NEAREST

    def __init__(self, obj=None, device=None, borrow=False):
        self.__shared = _Shared(None)
        """
        ImageFilter
        """
        if isinstance(obj, ImageFilter):
            self.__shared = obj.__shared
            return
        """
        _C.ts_ImageFilter
        """
        if isinstance(obj, _C.POINTER(_C.ts_ImageFilter)):
            self.__shared = _Shared(obj, None if borrow else _C.ts_free_ImageFilter)
            return

        if obj is not None:
            raise Exception("argument {}: expected ImageFilter or POINTER(ts_ImageFilter) instance instead of {}".
                            format(1, type(obj).__name__))

        if not isinstance(device, (Device, _C.POINTER(_C.ts_Device))):
            raise Exception("argument {}: expected Device or POINTER(ts_Device) instance instead of {}".
                            format(2, type(device).__name__))

        """
        new ImageFilter
        """
        c_image_filter = _C.ts_new_ImageFilter(device)
        _C.ts_api_check_pointer(c_image_filter)
        self.__shared = _Shared(c_image_filter, None if borrow else _C.ts_free_ImageFilter)

    def dispose(self):
        self.__shared.dispose()

    @property
    def _as_parameter_(self):
        return self.__shared.raw

    @property
    def raw(self):
        return self._as_parameter_

    def clear(self):
        _C.ts_api_check_bool(_C.ts_ImageFilter_clear(self))

    def compile(self):
        _C.ts_api_check_bool(_C.ts_ImageFilter_compile(self))

    def to_float(self):
        _C.ts_api_check_bool(_C.ts_ImageFilter_to_float(self))

    def scale(self, scale):
        # type: (float) -> None
        _C.ts_api_check_bool(_C.ts_ImageFilter_scale(self, scale))

    def sub_mean(self, mean):
        # type: (float) -> None
        c_array, c_len, _ = _to_ctypes_array(mean, _C.c_float)
        _C.ts_api_check_bool(_C.ts_ImageFilter_sub_mean(self, c_array, c_len))

    def div_std(self, std):
        # type: (float) -> None
        c_array, c_len, _ = _to_ctypes_array(std, _C.c_float)
        _C.ts_api_check_bool(_C.ts_ImageFilter_div_std(self, c_array, c_len))

    def resize(self, width, height=None, method=None):
        # type: (int, int, int) -> None
        assert method is None or method in {ResizeMethod.BICUBIC, ResizeMethod.BILINEAR, ResizeMethod.NEAREST}
        if method is not None:
            if height:
                _C.ts_api_check_bool(_C.ts_ImageFilter_resize_v2(self, width, height, method))
            else:
                _C.ts_api_check_bool(_C.ts_ImageFilter_resize_scalar_v2(self, width, method))
        else:
            if height:
                _C.ts_api_check_bool(_C.ts_ImageFilter_resize(self, width, height))
            else:
                _C.ts_api_check_bool(_C.ts_ImageFilter_resize_scalar(self, width))

    def center_crop(self, width, height=None):
        # type: (int, int) -> None
        if height is None:
            height = width
        _C.ts_api_check_bool(_C.ts_ImageFilter_center_crop(self, width, height))

    def channel_swap(self, shuffle):
        # type: (List[int]) -> None
        c_array, c_len, _ = _to_ctypes_array(shuffle, _C.c_int32)
        _C.ts_api_check_bool(_C.ts_ImageFilter_channel_swap(self, c_array, c_len))

    def to_chw(self):
        _C.ts_api_check_bool(_C.ts_ImageFilter_to_chw(self))

    def prewhiten(self):
        _C.ts_api_check_bool(_C.ts_ImageFilter_prewhiten(self))

    def letterbox(self, width, height, outer_value=0, method=None):
        # type: (int, int, float, int) -> None
        assert method is None or method in {ResizeMethod.BICUBIC, ResizeMethod.BILINEAR, ResizeMethod.NEAREST}
        if method is not None:
            _C.ts_api_check_bool(_C.ts_ImageFilter_letterbox_v2(self, width, height, outer_value, method))
        else:
            _C.ts_api_check_bool(_C.ts_ImageFilter_letterbox(self, width, height, outer_value))

    def divided(self, width, height, padding_value=0):
        # type: (int, int, float) -> None
        _C.ts_api_check_bool(_C.ts_ImageFilter_divided(self, width, height, padding_value))

    def run(self, x):
        # type: (Tensor) -> Tensor
        if not isinstance(x, (Tensor, _C.POINTER(_C.ts_Tensor))):
            raise Exception("argument {}: expected Tensor or POINTER(ts_Tensor) instance instead of {}".
                            format(1, type(x).__name__))
        y = _C.ts_ImageFilter_run(self, x)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    def force_color(self):
        _C.ts_api_check_bool(_C.ts_ImageFilter_force_color(self))

    def force_gray(self, scale=None):
        # type: (List[float]) -> None
        if scale is None:
            _C.ts_api_check_bool(_C.ts_ImageFilter_force_gray(self))
        else:
            c_array, c_len, _ = _to_ctypes_array(scale, _C.c_float)
            _C.ts_api_check_bool(_C.ts_ImageFilter_force_gray_v2(self, c_array, c_len))

    def force_bgr2gray(self):
        return self.force_gray([0.114, 0.587, 0.299])

    def force_rgb2gray(self):
        return self.force_gray([0.299, 0.587, 0.114])

    def norm_image(self, epsilon):
        # type: (float) -> None
        _C.ts_api_check_bool(_C.ts_ImageFilter_norm_image(self, epsilon))

    def module(self):
        # type: () -> Module
        module = _C.ts_ImageFilter_module(self)
        _C.ts_api_check_pointer(module)
        return Module(module)


ResizeMethod = ImageFilter.ResizeMethod


class Workbench(object):
    def __init__(self, obj=None, device=None, borrow=False):
        self.__shared = _Shared(None)
        """
        Workbench
        """
        if isinstance(obj, Workbench):
            self.__shared = obj.__shared
            return
        """
        _C.ts_Workbench
        """
        if isinstance(obj, _C.POINTER(_C.ts_Workbench)):
            self.__shared = _Shared(obj, None if borrow else _C.ts_free_Workbench)
            return

        if obj is not None:
            raise Exception("argument {}: expected Workbench or POINTER(ts_Workbench) instance instead of {}".
                            format(1, type(obj).__name__))

        if not isinstance(device, (Device, _C.POINTER(_C.ts_Device))):
            raise Exception("argument {}: expected Device or POINTER(ts_Device) instance instead of {}".
                            format(2, type(device).__name__))

        """
        new Workbench
        """
        c_workbench = _C.ts_new_Workbench(device)
        _C.ts_api_check_pointer(c_workbench)
        self.__shared = _Shared(c_workbench, None if borrow else _C.ts_free_Workbench)

    def dispose(self):
        self.__shared.dispose()

    @property
    def _as_parameter_(self):
        return self.__shared.raw

    @property
    def raw(self):
        return self._as_parameter_

    @staticmethod
    def Load(module, device, options=None):
        # type: (Module, Device, str) -> Workbench
        if not isinstance(module, (Module, _C.POINTER(_C.ts_Module))):
            raise Exception("argument {}: expected Module or POINTER(ts_Module) instance instead of {}".
                            format(1, type(module).__name__))
        if not isinstance(device, (Device, _C.POINTER(_C.ts_Device))):
            raise Exception("argument {}: expected Device or POINTER(ts_Device) instance instead of {}".
                            format(2, type(device).__name__))
        options = _compatible_string(options)
        if options is not None and not isinstance(options, str):
            raise Exception("argument {}: expected None or str instance instead of {}".
                            format(3, type(options).__name__))
        c_workbench = _C.POINTER(_C.ts_Workbench)
        if isinstance(options, str):
            options = options.encode()
            c_workbench = _C.ts_Workbench_Load_v2(module, device, options)
        else:
            c_workbench = _C.ts_Workbench_Load(module, device)
        _C.ts_api_check_pointer(c_workbench)
        return Workbench(c_workbench)

    def clone(self):
        # type: () -> Workbench
        x = _C.ts_Workbench_clone(self)
        _C.ts_api_check_pointer(x)
        return Workbench(x)

    def copy(self):
        # type: () -> Workbench
        return self.clone()

    def input(self, slot, tensor):
        # type: (Union[int, str], Tensor) -> None
        tensor = Tensor(tensor)
        if isinstance(slot, int):
            _C.ts_api_check_bool(_C.ts_Workbench_input(self, slot, tensor))
            return
        slot = _compatible_string(slot)
        if isinstance(slot, str):
            slot = slot.encode()
            _C.ts_api_check_bool(_C.ts_Workbench_input_by_name(self, slot, tensor))
            return
        raise Exception("argument {}: expected int or str instance instead of {}".
                        format(1, type(slot).__name__))

    def run(self):
        # type: () -> None
        _C.ts_api_check_bool(_C.ts_Workbench_run(self))

    def run_hook(self, node_names):
        # type: (List[str]) -> None
        if not isinstance(node_names, (list, tuple)):
            node_names = [node_names, ]
        node_names = [node_name.encode() for node_name in node_names]
        c_node_names = (_C.c_char_p * len(node_names))(*node_names)
        c_len = _C.c_int32(len(node_names))
        _C.ts_api_check_bool(_C.ts_Workbench_run_hook(self, c_node_names, c_len))

    def output(self, slot, tensor=None):
        # type: (Union[int, str], Tensor) -> Tensor
        if tensor is not None and not isinstance(tensor, (Tensor, _C.POINTER(_C.ts_Tensor))):
            raise Exception("argument {}: expected Tensor or POINTER(ts_Tensor) instance instead of {}".
                            format(2, type(tensor).__name__))
        if tensor is None:
            tensor = Tensor(dtype=VOID)
        if isinstance(slot, int):
            _C.ts_api_check_bool(_C.ts_Workbench_output(self, slot, tensor))
            return tensor
        slot = _compatible_string(slot)
        if isinstance(slot, str):
            slot = slot.encode()
            _C.ts_api_check_bool(_C.ts_Workbench_output_by_name(self, slot, tensor))
            return tensor
        raise Exception("argument {}: expected int or str instance instead of {}".
                        format(1, type(slot).__name__))

    def set_computing_thread_number(self, number):
        # type: (int) -> None
        _C.ts_api_check_bool(_C.ts_Workbench_set_computing_thread_number(self, number))

    def bind_filter(self, slot, filter):
        # type: (Union[int, str], ImageFilter) -> None
        if not isinstance(filter, (ImageFilter, _C.POINTER(_C.ts_ImageFilter))):
            raise Exception("argument {}: expected ImageFilter or POINTER(ts_ImageFilter) instance instead of {}".
                            format(2, type(filter).__name__))
        if isinstance(slot, int):
            _C.ts_api_check_bool(_C.ts_Workbench_bind_filter(self, slot, filter))
            return
        slot = _compatible_string(slot)
        if isinstance(slot, str):
            slot = slot.encode()
            _C.ts_api_check_bool(_C.ts_Workbench_bind_filter_by_name(self, slot, filter))
            return
        raise Exception("argument {}: expected int or str instance instead of {}".
                        format(1, type(slot).__name__))

    def setup(self, program):
        # type: (Program) -> None
        if not isinstance(program, (Program, _C.POINTER(_C.ts_Program))):
            raise Exception("argument {}: expected Program or POINTER(ts_Program) instance instead of {}".
                            format(1, type(program).__name__))
        _C.ts_api_check_bool(_C.ts_Workbench_setup(self, program))

    def setup_context(self):
        # type: () -> None
        _C.ts_api_check_bool(_C.ts_Workbench_setup_context(self))

    def compile(self, module, options=None):
        # type: (Module, str) -> Program
        if not isinstance(module, (Module, _C.POINTER(_C.ts_Module))):
            raise Exception("argument {}: expected Module or POINTER(ts_Module) instance instead of {}".
                            format(1, type(module).__name__))
        options = _compatible_string(options)
        if options is not None and not isinstance(options, str):
            raise Exception("argument {}: expected None or str instance instead of {}".
                            format(2, type(options).__name__))
        c_program = _C.POINTER(_C.ts_Program)
        if isinstance(options, str):
            options = options.encode()
            c_program = _C.ts_Workbench_compile_v2(self, module, options)
        else:
            c_program = _C.ts_Workbench_compile(self, module)
        _C.ts_api_check_pointer(c_program)
        return Program(c_program)

    def setup_device(self):
        # type: () -> None
        _C.ts_api_check_bool(_C.ts_Workbench_setup_device(self))

    def setup_runtime(self):
        # type: () -> None
        _C.ts_api_check_bool(_C.ts_Workbench_setup_runtime(self))

    def input_count(self):
        # type: () -> int
        return _C.ts_Workbench_input_count(self)

    def output_count(self):
        # type: () -> int
        return _C.ts_Workbench_output_count(self)

    def set_operator_param(self, node_name, param, value):
        # type: (str, str, Tensor) -> None
        node_name = node_name.encode()
        param = param.encode()
        value = Tensor(value)
        _C.ts_api_check_bool(_C.ts_Workbench_set_operator_param(self, node_name, param, value))

    def summary(self):
        # type: () -> str
        s = _C.ts_Workbench_summary(self)
        _C.ts_api_check_pointer(s)
        return str(s.decode())


class OperatorParams(object):
    def __init__(self, obj=None, borrow=False):
        self.__shared = _Shared(None)
        """
        OperatorParams
        """
        if isinstance(obj, OperatorParams):
            self.__shared = obj.__shared
            return
        """
        _C.ts_OperatorParams
        """
        if isinstance(obj, _C.POINTER(_C.ts_OperatorParams)):
            if not borrow:
                raise NotImplementedError("No free function given in _C")
            self.__shared = _Shared(obj, None)
            return

        if obj is not None:
            raise Exception("argument {}: expected OperatorParams or POINTER(ts_OperatorParams) instance instead of {}".
                            format(1, type(obj).__name__))

    def dispose(self):
        self.__shared.dispose()

    @property
    def _as_parameter_(self):
        return self.__shared.raw

    @property
    def raw(self):
        return self._as_parameter_

    def get(self, item):
        # type: (str) -> Tensor
        item = item.encode()
        x = _C.ts_OperatorParams_get(self, item)
        return Tensor(x)

    def has(self, item):
        # type: (str) -> bool
        x = self.get(item)
        y = bool(x)
        del x
        return y

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        return self.has(item)


class OperatorContext(object):
    def __init__(self, obj=None, borrow=False):
        self.__shared = _Shared(None)
        """
        OperatorContext
        """
        if isinstance(obj, OperatorContext):
            self.__shared = obj.__shared
            return
        """
        _C.ts_OperatorContext
        """
        if isinstance(obj, _C.POINTER(_C.ts_OperatorContext)):
            if not borrow:
                raise NotImplementedError("No free function given in _C")
            self.__shared = _Shared(obj, None)
            return

        if obj is not None:
            raise Exception("argument {}: expected OperatorContext or POINTER(ts_OperatorContext) instance instead of {}".
                            format(1, type(obj).__name__))

    def dispose(self):
        self.__shared.dispose()

    @property
    def _as_parameter_(self):
        return self.__shared.raw

    @property
    def raw(self):
        return self._as_parameter_


class Operator(object):
    def __init__(self):
        pass

    def dispose(self):
        # type: () -> None
        """
        Will called in Operator_free
        :return:
        """
        pass

    def init(self, params, context):
        # type: (OperatorParams, OperatorContext) -> None
        """
        :param params:
        :param context:
        :return: None
        """
        pass

    def infer(self, args, context):
        # type: (List[Tensor], OperatorContext) -> List[tuple]
        """
        :param args:
        :param context:
        :return: list of tuple like [(FLOAT32, (1, 3, 4, 4)), (INT32, (1, 2))]
        """
        raise NotImplementedError

    def run(self, args, context):
        # type: (List[Tensor], OperatorContext) -> Union[Tensor, List[Tensor]]
        """
        :param args:
        :param context:
        :return: list of tuple like [(FLOAT32, (1, 3, 4, 4)), (INT32, (1, 2))]
        """
        raise NotImplementedError


_RegisterOperator = {}
_OperatorPool = {}


def RegisterOperator(cls, device, op):
    # type: (type, str, str) -> None
    if not isinstance(cls, type) and not issubclass(cls, Operator):
        raise Exception("cls must be the subclass of Operator")

    def operator_new():
        try:
            pyobj = _C.py_object(cls())
            p_pyobj = _C.pointer(pyobj)
            p_pyobj = _C.cast(p_pyobj, _C.c_void_p)
            c_void_ptr = p_pyobj.value
            _OperatorPool[c_void_ptr] = pyobj
            return c_void_ptr
        except Exception as e:
            import traceback
            message = traceback.format_exc()
            set_error_message(message=message)
        return None

    def operator_free(obj):
        try:
            if bool(obj):
                obj = _C.cast(obj, _C.c_void_p)
                c_void_ptr = obj.value

                if c_void_ptr in _OperatorPool:
                    del _OperatorPool[c_void_ptr]

                p_pyobj = _C.cast(obj, _C.POINTER(_C.py_object))
                obj = p_pyobj.contents.value

                if isinstance(obj, Operator):
                    obj.dispose()
            return
        except Exception as e:
            import traceback
            message = traceback.format_exc()
            sys.stderr.write("{}\n".format(message))
            set_error_message(message=message)
        return

    def operator_init_ex(obj, params, context):
        try:
            obj = _C.cast(obj, _C.c_void_p)
            c_void_ptr = obj.value
            assert c_void_ptr in _OperatorPool
            p_pyobj = _C.cast(obj, _C.POINTER(_C.py_object))
            obj = p_pyobj.contents.value
            if isinstance(obj, Operator):
                obj.init(OperatorParams(params, borrow=True), OperatorContext(context, borrow=True))
                return 1
            else:
                raise Exception("argument {}: expected py_object instance instead of {}".
                                format(1, type(obj).__name__))
        except Exception as e:
            import traceback
            message = traceback.format_exc()
            set_error_message(message=message)
        return 0

    def operator_run(obj, argc, argv, context):
        try:
            obj = _C.cast(obj, _C.c_void_p)
            c_void_ptr = obj.value
            assert c_void_ptr in _OperatorPool
            p_pyobj = _C.cast(obj, _C.POINTER(_C.py_object))
            obj = p_pyobj.contents.value
            if isinstance(obj, Operator):
                argc = int(argc)
                args = [Tensor(argv[i], borrow=True) for i in range(argc)]
                out = obj.run(args, OperatorContext(context, borrow=True))
                if not isinstance(out, (list, tuple)):
                    out = [out]
                out = [Tensor(t) for t in out]
                out = Tensor.Pack(out)
                x = out.release()
                x = _C.cast(x, _C.c_void_p)
                return x.value
            else:
                raise Exception("argument {}: expected py_object instance instead of {}".
                                format(1, type(obj).__name__))
        except Exception as e:
            import traceback
            message = traceback.format_exc()
            set_error_message(message=message)
        return None

    def operator_infer(obj, argc, argv, context):
        try:
            obj = _C.cast(obj, _C.c_void_p)
            c_void_ptr = obj.value
            assert c_void_ptr in _OperatorPool
            p_pyobj = _C.cast(obj, _C.POINTER(_C.py_object))
            obj = p_pyobj.contents.value
            if isinstance(obj, Operator):
                proto = []
                if obj.infer == Operator.infer: # no infer override, call run instead
                    out = operator_run(obj, argc, argv, context)
                    if not out:
                        return None
                    out = Tensor(out, borrow=False)
                    proto = [(field.dtype, field.shape) for field in out]
                else:
                    argc = int(argc)
                    args = [Tensor(argv[i], borrow=True) for i in range(argc)]
                    proto = obj.infer(args, OperatorContext(context, borrow=True))
                packed = [len(proto)]
                for p in proto:
                    packed.append(p[0])
                    packed.append(len(p[1]))
                    packed.extend(p[1])
                x = Tensor(obj=packed, dtype=INT32).release()
                x = _C.cast(x, _C.c_void_p)
                return x.value
            else:
                raise Exception("argument {}: expected py_object instance instead of {}".
                                format(1, type(obj).__name__))
        except Exception as e:
            import traceback
            message = traceback.format_exc()
            set_error_message(message=message)
        return None

    operator_new = _C.ts_new_Operator(operator_new)
    operator_free = _C.ts_free_Operator(operator_free)
    operator_init_ex = _C.ts_Operator_init_ex(operator_init_ex)
    operator_infer = _C.ts_Operator_infer(operator_infer)
    operator_run = _C.ts_Operator_run(operator_run)

    _RegisterOperator[(device, op)] = (operator_new, operator_free, operator_init_ex, operator_infer, operator_run)

    device = device.encode()
    op = op.encode()
    _C.ts_Operator_RegisterEx(device, op,
                              operator_new,
                              operator_free,
                              operator_init_ex,
                              operator_infer,
                              operator_run)


class intime(object):
    @staticmethod
    def transpose(x, shuffle):
        x = Tensor(x)
        c_array, c_len, _ = _to_ctypes_array(shuffle, INT32)
        y = _C.ts_intime_transpose(x, c_array, c_len)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def sigmoid(x,):
        x = Tensor(x)
        y = _C.ts_intime_sigmoid(x)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def gather(x, indices):
        x = Tensor(x)
        indices = Tensor(indices, dtype=INT32)
        y = _C.ts_intime_gather(x, indices)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def concat(tensors, dim):
        tensors = [Tensor(tensor) for tensor in tensors]
        c_tensors = [tensor.raw for tensor in tensors]
        c_len = len(tensors)
        c_tensors = (_C.POINTER(_C.ts_Tensor) * c_len)(*c_tensors)
        y = _C.ts_intime_concat(c_tensors, c_len, dim)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def softmax(x, dim, smooth=True):
        x = Tensor(x)
        smooth = 1 if smooth else 0
        y = _C.ts_intime_softmax(x, dim, smooth)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def pad(x, padding, padding_value=0):
        x = Tensor(x)
        padding = Tensor(padding, dtype=INT32)
        y = _C.ts_intime_pad(x, padding, padding_value)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def cast(x, dtype):
        x = Tensor(x)
        dtype = DC.to_ts_dtype(dtype=dtype)
        y = _C.ts_intime_cast(x, dtype)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def resize2d(x, size, method=ResizeMethod.BILINEAR):
        x = Tensor(x)
        size = Tensor(size, dtype=INT32)
        assert method is None or method in {ResizeMethod.BICUBIC, ResizeMethod.BILINEAR, ResizeMethod.NEAREST}
        y = _C.ts_intime_resize2d(x, size, method)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def affine_sample2d(x, size, affine, dim, outer_value=0, method=ResizeMethod.BILINEAR):
        x = Tensor(x)
        size = Tensor(size, dtype=INT32)
        affine = Tensor(affine, dtype=FLOAT32)
        assert method is None or method in {ResizeMethod.BICUBIC, ResizeMethod.BILINEAR, ResizeMethod.NEAREST}
        y = _C.ts_intime_affine_sample2d(x, size, affine, dim, outer_value, method)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def affine_on_sample2d(x, size, affine, dim, method=ResizeMethod.BILINEAR):
        x = Tensor(x)
        size = Tensor(size, dtype=INT32)
        affine = Tensor(affine, dtype=FLOAT32)
        assert method is None or method in {ResizeMethod.BICUBIC, ResizeMethod.BILINEAR, ResizeMethod.NEAREST}
        y = _C.ts_intime_affine_on_sample2d(x, size, affine, dim, method)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

    @staticmethod
    def memcpy(dst_desc, dst_shift, src_desc, src_shift, size,
               dst_ptr=None, src_ptr=None):
        # type: (Tenosr, object, int, Tensor, object, int, int) -> None
        if not isinstance(dst_desc, Tensor):
            raise RuntimeError("argument {}: expected Tensor instance instead of {}".
                               format(1, type(dst_desc).__name__))
        if not isinstance(src_desc, Tensor):
            raise RuntimeError("argument {}: expected Tensor instance instead of {}".
                               format(2, type(src_desc).__name__))

        if dst_ptr is not None:
            raise NotImplementedError("dst_ptr = {}".format(type(dst_ptr).__name__))
        if src_ptr is not None:
            raise NotImplementedError("src_ptr = {}".format(type(src_ptr).__name__))

        dst_shift = _C.c_int64(dst_shift)
        src_shift = _C.c_int64(src_shift)
        size = _C.c_int64(size)

        copied = _C.ts_intime_memcpy(
            dst_desc, dst_ptr, dst_shift,
            src_desc, src_ptr, src_shift,
            size)

        return copied

    @staticmethod
    def matmul(A, B, transpose=False):
        A = Tensor(A)
        B = Tensor(B)
        transpose = 1 if transpose else 0
        y = _C.ts_intime_matmul(A, B, transpose)
        _C.ts_api_check_pointer(y)
        return Tensor(y)

