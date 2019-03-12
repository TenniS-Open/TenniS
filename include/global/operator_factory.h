//
// Created by kier on 2018/6/29.
//

#ifndef TENSORSTACK_OPERATOR_FACTORY_H
#define TENSORSTACK_OPERATOR_FACTORY_H

#include <functional>
#include "runtime/operator.h"
#include "utils/static.h"

namespace ts {

    class TS_DEBUG_API OperatorCreator {
    public:

        using function =  std::function<Operator::shared(void)>;

        Operator::shared OperatorCreatorFunction();

        static function Query(const DeviceType &device_type,
                              const std::string &operator_name) TS_NOEXCEPT;

        static void Register(const DeviceType &device_type,
                             const std::string &operator_name,
                             const function &operator_creator) TS_NOEXCEPT;

        static Operator::shared Create(const DeviceType &device_type,
                                       const std::string &operator_name,
                                       bool strict = false) TS_NOEXCEPT;

        static function Query(const DeviceType &device_type,
                              const std::string &operator_name, bool strict) TS_NOEXCEPT;

        static void Clear();
    };
}

/**
 * Static action
 */
#define TS_REGISTER_OPERATOR(CLASS_NAME, DEVICE_TYPE, OP_NAME) \
    namespace \
    { \
        static ts::Operator::shared _ts_concat_name(CLASS_NAME, _CREATOR)() { return std::make_shared<CLASS_NAME>(); } \
        ts::StaticAction ts_serial_name(_ts_static_action_)(ts::OperatorCreator::Register, DEVICE_TYPE, OP_NAME, _ts_concat_name(CLASS_NAME, _CREATOR)); \
    }


#endif //TENSORSTACK_OPERATOR_FACTORY_H
