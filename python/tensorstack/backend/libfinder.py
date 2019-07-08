#!/usr/bin/env python

"""
:author Kier
"""

import platform
import os
import sys
from ctypes.util import find_library
from ctypes import CDLL


def safeCDLL(libpath):
    try:
        lib = CDLL(libpath)
        return lib, ""
    except Exception as e:
        return None, str(e)


def load_library(libname):
    """
    :param libname:
    :return: lib, msg
    if lib is None and msg is None, No file found
    if lib is None and msg is str, there is exception when open library
    """
    # try load lib form file
    CWD = os.getcwd()
    CFD = os.path.abspath(os.path.dirname(__file__))
    PATH = os.environ["PATH"] if "PATH" in os.environ else ""
    LIBRARTY_PATH = os.environ["LIBRARTY_PATH"] if "LIBRARTY_PATH" in os.environ else ""
    LD_LIBRARY_PATH = os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ else ""
    DYLD_LIBRARY_PATH = os.environ["DYLD_LIBRARY_PATH"] if "DYLD_LIBRARY_PATH" in os.environ else ""
    SYS_PATH = type(sys.path)(sys.path)

    sep = ";" if platform.system() == "Windows" else ":"

    ALL_PATH = [CWD, CFD]
    ALL_PATH += LIBRARTY_PATH.split(sep)
    ALL_PATH += LD_LIBRARY_PATH.split(sep)
    ALL_PATH += DYLD_LIBRARY_PATH.split(sep)
    ALL_PATH += PATH.split(sep)
    ALL_PATH += SYS_PATH

    ALL_PATH_ENV = sep.join(ALL_PATH)

    os.environ["PATH"] = ALL_PATH_ENV
    os.environ["LIBRARTY_PATH"] = ALL_PATH_ENV
    os.environ["LD_LIBRARY_PATH"] = ALL_PATH_ENV
    os.environ["DYLD_LIBRARY_PATH"] = ALL_PATH_ENV
    sys.path = ALL_PATH

    lib = None
    msg = None
    libpath = find_library(libname)
    if libpath is None:
        libpath = "%s.dll" % libname if platform.system() == "Windows" else "lib%s.so" % libname
    if libpath.find("/") >= 0 or libpath.find("\\") >= 0:
        sys.stderr.write("Find {} in: {}\n".format(libname, libpath))
        lib, msg = safeCDLL(libpath)
    else:
        for root in ALL_PATH:
            fullpath = os.path.join(root, libpath)
            if not os.path.isfile(fullpath): continue
            lib, msg = safeCDLL(fullpath)
            if lib is not None:
                sys.stderr.write("Find {} in: {}\n".format(libname, fullpath))
                msg = None
                break
            else:
                break
                # sys.stderr.write("{}\n".format(msg))

    os.environ["PATH"] = PATH
    os.environ["LIBRARTY_PATH"] = LIBRARTY_PATH
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    os.environ["DYLD_LIBRARY_PATH"] = DYLD_LIBRARY_PATH
    sys.path = SYS_PATH

    return lib, msg
