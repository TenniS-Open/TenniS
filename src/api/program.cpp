//
// Created by kier on 2019/5/7.
//

#include <api/program.h>

#include "declare_module.h"
#include "declare_tensor.h"
#include "declare_program.h"

using namespace ts;

ts_Program *ts_Program_Compile(const ts_Module *module, const ts_Device *device) {
    TRY_HEAD
    if (!module) throw Exception("NullPointerException: @param: 1");
    if (!device) throw Exception("NullPointerException: @param: 2");
    std::unique_ptr<ts_Program> program(new ts_Program(
            Program::Compile(module->pointer, ComputingDevice(device->type, device->id))));
    RETURN_OR_CATCH(program.release(), nullptr)
}

void ts_free_Program(const ts_Program *program) {
    TRY_HEAD
    delete program;
    TRY_TAIL
}

ts_Program *ts_Program_clone(ts_Program *program) {
    TRY_HEAD
        if (!program) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Program> dolly(new ts_Program((*program)->clone()));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}
