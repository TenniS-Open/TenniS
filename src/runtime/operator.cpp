//
// Created by seeta on 2018/5/19.
//

#include "stack/operator.h"

namespace ts {

    void Operator::set(const std::string &param, const Tensor &value) {
        this->m_params.insert(std::make_pair(param, value));
    }

    Tensor &Operator::get(const std::string &param) {
        return this->m_params.at(param);
    }

    const Tensor &Operator::get(const std::string &param) const {
        return this->m_params.at(param);
    }

    void Operator::clear(const std::string &param) {
        this->m_params.erase(param);
    }

    void Operator::clear_params() {
        this->m_params.clear();
    }
}