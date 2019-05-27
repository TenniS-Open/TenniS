#!python
# coding: UTF-8
"""
author: kier
"""

import re


class Session(object):
    """
    Dict of key values
    """
    def __init__(self, name=None):
        if name is None:
            name = ""
        self.__name = name
        self.__options = {}

    @property
    def name(self):
        return self.__name

    def __setitem__(self, key, value):
        self.__options.__setitem__(key, value)

    def __getitem__(self, item):
        return self.__options.__getitem__(item)

    def __len__(self):
        return self.__options.__len__()

    def __repr__(self):
        lines = ["[{}]".format(self.__name)]
        for k, v in self.__options.iteritems():
            lines.append("{} = {}".format(k, v))
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()

    def has(self, key):
        return key in self.__options

    def __get_class(self, cls, key, value=None):
        if key in self.__options:
            return cls(self.__options[key].strip())
        if value is None:
            return cls()
        return value

    def __get_class_list(self, cls, key, value=None, sep=','):
        if key in self.__options:
            values = self.__options[key]
            values = values.split(sep)
            values = [cls(s.strip()) for s in values]
            return values
        if value is None:
            return []
        return value

    def get_int(self, key, value):
        return self.__get_class(int, key, value)

    def get_float(self, key, value):
        return self.__get_class(float, key, value)

    def get_string(self, key, value):
        return self.__get_class(str, key, value)

    def get_int_list(self, key, value=None, sep=','):
        return self.__get_class_list(int, key, value, sep=sep)

    def get_float_list(self, key, value=None, sep=','):
        return self.__get_class_list(float, key, value, sep=sep)

    def get_string_list(self, key, value=None, sep=','):
        return self.__get_class_list(str, key, value, sep=sep)


class Config(object):
    """
    List of Session
    """
    def __init__(self):
        self.__sessions = []
        self.__default_session = Session()

    def __getitem__(self, item):
        if item is None:
            return self.__default_session
        return self.__sessions[item]

    def __len__(self):
        return self.__sessions.__len__()

    def read(self, path):
        self.__init__()
        current_session = self.__default_session

        regex_annotation = re.compile(r'#.*')
        regex_session = re.compile(r'\[(.*)\]')
        regex_option = re.compile(r'([^=]*)=(.*)')

        with open(path, 'r') as file:
            for line in file.readlines():
                line = str.strip(line)
                if len(line) == 0:
                    continue
                match_annotation = re.match(regex_annotation, line)
                if match_annotation is not None:
                    continue
                match_session = re.match(regex_session, line)
                if match_session is not None:
                    new_session = Session(match_session.group(1))
                    self.__sessions.append(new_session)
                    current_session = new_session
                    continue
                match_option = re.match(regex_option, line)
                if match_option is not None:
                    key = match_option.group(1).strip()
                    value = match_option.group(2).strip()
                    current_session.__setitem__(key, value)
                    continue
                raise Exception("cfg format error, got line: {}".format(line))

    def __repr__(self):
        lines = []
        for session in self.__sessions:
            lines.append(repr(session))
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


def option_find_int(session, key, value):
    # type: (Session, str, int) -> int
    return session.get_int(key, value)


def option_find_float(session, key, value):
    # type: (Session, str, float) -> float
    return session.get_float(key, value)


def option_find_str(session, key, value):
    # type: (Session, str, str) -> str
    return session.get_string(key, value)


def option_find_int_list(session, key, value):
    # type: (Session, str, list) -> list
    return session.get_int_list(key, value)


def option_find_float_list(session, key, value):
    # type: (Session, str, list) -> list
    return session.get_float_list(key, value)


def option_find_str_list(session, key, value):
    # type: (Session, str, list) -> list
    return session.get_string_list(key, value)


option_find_int_quiet = option_find_int
option_find_float_quiet = option_find_float
option_find_str_quiet = option_find_str


def read_config(path):
    cfg = Config()
    cfg.read(path)
    return cfg
