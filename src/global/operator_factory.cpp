//
// Created by kier on 2018/6/29.
//

#include "global/operator_factory.h"
#include <map>
#include <global/operator_factory.h>

#include "global/memory_device.h"


namespace ts {
    using Name = std::pair<DeviceType, std::string>;

    template<typename K, typename V>
    using map = std::map<K, V>;

    static map<Name, OperatorCreator::function> &MapNameCreator() {
        static map<Name, OperatorCreator::function> map_name_creator;
        return map_name_creator;
    };

    OperatorCreator::function OperatorCreator::Query(const DeviceType &device_type,
                                                     const std::string &operator_name) TS_NOEXCEPT {
        auto &map_name_creator = MapNameCreator();
        Name device_operator_name = std::make_pair(device_type, operator_name);
        auto name_creator = map_name_creator.find(device_operator_name);
        if (name_creator != map_name_creator.end()) {
            return name_creator->second;
        }
        return OperatorCreator::function(nullptr);
    }

    void OperatorCreator::Register(const DeviceType &device_type,
                                   const std::string &operator_name,
                                   const OperatorCreator::function &operator_creator) TS_NOEXCEPT {
        auto &map_name_creator = MapNameCreator();
        Name device_operator_name = std::make_pair(device_type, operator_name);
        map_name_creator[device_operator_name] = operator_creator;
    }

    void OperatorCreator::Clear() {
        auto &map_name_creator = MapNameCreator();
        map_name_creator.clear();
    }

    static OperatorCreator::function TalentQuery(const DeviceType &device_type,
                                                 const std::string &operator_name, bool strict) {
        // step 1: check in strict mode
        auto creator = OperatorCreator::Query(device_type, operator_name);

        if (strict) return creator;

        if (creator == nullptr) {
            // step 2.x: try find operator on memory device, if computing device failed
            auto memory_device = ComputingMemory::Query(device_type);
            creator = OperatorCreator::Query(memory_device, operator_name);
        }

        if (creator == nullptr) {
            // step 2.y: try find operator on CPU version
            if (device_type != CPU) {
                creator = OperatorCreator::Query(CPU, operator_name);
            }
        }

        return creator;
    }

    Operator::shared OperatorCreator::CreateNoException(const DeviceType &device_type, const std::string &operator_name,
                                                        bool strict) TS_NOEXCEPT {
        auto creator = TalentQuery(device_type, operator_name, strict);
        if (!creator) return nullptr;
        return creator();
    }

    OperatorCreator::function
    OperatorCreator::Query(const DeviceType &device_type, const std::string &operator_name, bool strict) TS_NOEXCEPT {
        return TalentQuery(device_type, operator_name, strict);
    }

    Operator::shared
    OperatorCreator::Create(const DeviceType &device_type, const std::string &operator_name, bool strict)  {
        auto op = CreateNoException(device_type, operator_name, strict);
        if (op == nullptr) throw OperatorNotFoundException(device_type, operator_name);
        return op;
    }
}
