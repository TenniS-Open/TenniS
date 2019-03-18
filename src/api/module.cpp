//
// Created by kier on 2019/3/16.
//

#include "declare_module.h"

using namespace ts;

ts_Module *ts_Module_Load(const char *filename, ts_SerializationFormat format) {
    TRY_HEAD
    if (!filename) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_Module> module(new ts_Module(
            Module::Load(filename, Module::SerializationFormat(format))));
    RETURN_OR_CATCH(module.release(), nullptr)
}

void ts_free_Module(const ts_Module *module) {
    TRY_HEAD
    delete module;
    TRY_TAIL
}