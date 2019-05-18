//
// Created by kier on 19-5-18.
//

#ifndef TENSORSTACK_DECLARE_OPERATOR_H
#define TENSORSTACK_DECLARE_OPERATOR_H

#include <core/tensor_builder.h>
#include "declaration.h"

#include "api/operator.h"
#include "global/operator_factory.h"

#include "declare_tensor.h"

#include "runtime/stack.h"

struct ts_OperatorParams {
public:
    ts_OperatorParams(ts::Operator *op) : op(op) {}

    ts::Operator *op;
};

class APIPluginStack {
public:
    APIPluginStack(ts::Stack &stack) {
        int argc = stack.size();
        try {
            for (int i = 0; i < argc; ++i) {
                auto &tensor = stack[i];
                args.emplace_back(new ts_Tensor(tensor));
            }
        } catch (const ts::Exception &e) {
            for (auto &arg : args) delete arg;
            throw e;
        } catch (const std::exception &e) {
            for (auto &arg : args) delete arg;
            throw e;
        }
    }

    ~APIPluginStack() {
        for (auto &arg : args) {
            delete arg;
        }
    }

    std::vector<ts_Tensor *> args;
};

#endif //TENSORSTACK_DECLEARE_OPERATOR_H
