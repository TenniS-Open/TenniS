//
// Created by seeta on 2018/6/29.
//

#ifndef TENSORSTACK_OPERATOR_FACTORY_H
#define TENSORSTACK_OPERATOR_FACTORY_H

#include <functional>
#include "runtime/operator.h"
#include "utils/static.h"

namespace ts {
    using OperatorCreator =  std::function<Operator::shared(void)>;

    Operator::shared OperatorCreatorDeclaration();

    OperatorCreator QueryOperatorCreator(const std::string &operator_name) TS_NOEXCEPT;

    void RegisterOperatorCreator(const std::string &operator_name, const OperatorCreator &operator_creator) TS_NOEXCEPT;

    void ClearRegisteredOperatorCreator();
}

/**
 * Static action
 */
#define TS_REGISTER_OPERATOR(CLASS_NAME, OP_NAME) \
    namespace \
    { \
        static ts::Operator::shared _ts_concat_name(CLASS_NAME, _CREATOR)() { return std::make_shared<CLASS_NAME>(); } \
        ts::StaticAction ts_serial_name(_ts_static_action_)(ts::RegisterOperatorCreator, OP_NAME, _ts_concat_name(CLASS_NAME, _CREATOR)); \
    }


#endif //TENSORSTACK_OPERATOR_FACTORY_H
