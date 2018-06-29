//
// Created by seeta on 2018/6/29.
//

#include "global/operator_factory.h"

namespace ts {
    static std::map<std::string, OperatorCreator> &MapNameCreator() {
        static std::map<std::string, OperatorCreator> map_name_creator;
        return map_name_creator;
    };

    OperatorCreator QueryOperatorCreator(const std::string &operator_name) TS_NOEXCEPT {
        auto &map_name_creator = MapNameCreator();
        auto name_creator = map_name_creator.find(operator_name);
        if (name_creator != map_name_creator.end()) {
            return name_creator->second;
        }
        return OperatorCreator(nullptr);
    }

    void
    RegisterOperatorCreator(const std::string &operator_name, const OperatorCreator &operator_creator) TS_NOEXCEPT {
        auto &map_name_creator = MapNameCreator();
        map_name_creator.insert(std::make_pair(operator_name, operator_creator));
    }

    void ClearRegisteredOperatorCreator() {
        auto &map_name_creator = MapNameCreator();
        map_name_creator.clear();
    }
}
