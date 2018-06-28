//
// Created by seeta on 2018/5/19.
//

#include "stack/function.h"

namespace ts {

    void Function::set(const std::string &param, const Tensor &value) {
        this->m_params.insert(std::make_pair(param, value));
    }

    Tensor &Function::get(const std::string &param) {
        return this->m_params.at(param);
    }

    const Tensor &Function::get(const std::string &param) const {
        return this->m_params.at(param);
    }

    void Function::clear(const std::string &param) {
        this->m_params.erase(param);
    }

    void Function::clear_params() {
        this->m_params.clear();
    }
}