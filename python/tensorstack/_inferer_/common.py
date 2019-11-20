import numpy
from typing import Union, List, Tuple


class Form2D(object):
    def __init__(self, obj=None):
        # type: (Union[None, numpy.ndarray, List, Tuple, Form2D]) -> None
        self.height = 0
        self.width = 0
        if obj is None:
            return
        if isinstance(obj, Form2D):
            self.height = obj.height
            self.width = obj.width
        elif isinstance(obj, (numpy.ndarray, list, tuple)):
            obj = numpy.asarray(obj).reshape([-1])
            assert len(obj) == 1 or len(obj) == 2
            self.height = obj[0]
            self.width = obj[1] if len(obj) > 1 else obj[0]
        else:
            raise TypeError("Param 1 got not supported type: {}".format(type(obj)))

    def __str__(self):
        return "[{}, {}]".format(self.height, self.width)

    def __repr__(self):
        return self.__repr__()


class Aspect2D(object):
    def __init__(self, obj=None):
        # type: (Union[None, numpy.ndarray, List, Tuple, Aspect2D]) -> None
        self.top = 0
        self.bottom = 0
        self.left = 0
        self.right = 0
        if obj is None:
            return
        if isinstance(obj, Aspect2D):
            self.top = obj.top
            self.bottom = obj.bottom
            self.left = obj.left
            self.right = obj.right
        elif isinstance(obj, (numpy.ndarray, list, tuple)):
            obj = numpy.asarray(obj).reshape([-1])
            assert len(obj) == 1 or len(obj) == 2 or len(obj) == 4
            if len(obj) == 1:
                self.top = obj[0]
                self.bottom = obj[0]
                self.left = obj[0]
                self.right = obj[0]
            elif len(obj) == 2:
                self.top = obj[0]
                self.bottom = obj[0]
                self.left = obj[1]
                self.right = obj[1]
            elif len(obj) == 4:
                self.top = obj[0]
                self.bottom = obj[1]
                self.left = obj[2]
                self.right = obj[3]
        else:
            raise TypeError("Param 1 got not supported type: {}".format(type(obj)))

    def __str__(self):
        return "[[{}, {}], [{}, {}]]".format(self.top, self.bottom, self.left, self.right)

    def __repr__(self):
        return self.__repr__()


Padding2D = Aspect2D
Stride2D = Form2D
KSize2D = Form2D
Dilation2D = Form2D
Size2D = Form2D


def pooling2d_backward(y, padding, ksize, stride):
    # type: (Size2D, Padding2D, KSize2D, Stride2D) -> Size2D
    x = Size2D()
    x.height = (y.height - 1) * stride.height + ksize.height - padding.top - padding.bottom
    x.width = (y.width - 1) * stride.width + ksize.width - padding.left - padding.right
    return x


def conv2d_backward(y, padding, ksize, stride, dialations):
    # type: (Size2D, Padding2D, KSize2D, Stride2D, Dilation2D) -> Size2D
    x = Size2D()
    x.height = (y.height - 1) * stride.height + (dialations.height * (ksize.height - 1) + 1) \
           - padding.top - padding.bottom
    x.width = (y.width - 1) * stride.width + (dialations.width * (ksize.width - 1) + 1) \
          - padding.left - padding.right
    return x

