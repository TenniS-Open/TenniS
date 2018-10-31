//
// Created by seeta on 2018/5/19.
//

#include "runtime/operator.h"
#include "utils/box.h"

namespace ts {

    void Operator::set(const std::string &param, const Tensor &value) {
        if (!is_in_fields(param)) {
            throw Exception(
                    std::string("Unidentified param \"") + param + "\", did you mean \"" + fuzzy_field_name(param) +
                    "\"");
        }
        this->m_params.insert(std::make_pair(param, value));
    }

    Tensor &Operator::get(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            throw Exception(
                    std::string("Unidentified param \"") + param + "\", did you mean \"" + fuzzy_param_name(param) +
                    "\"");
        }
        return param_it->second;
    }

    const Tensor &Operator::get(const std::string &param) const {
        return const_cast<self *>(this)->get(param);
    }

    void Operator::clear(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            throw Exception(
                    std::string("Unidentified param \"") + param + "\", did you mean \"" + fuzzy_param_name(param) +
                    "\"");
        }
        this->m_params.erase(param_it);
    }

    void Operator::clear_params() {
        this->m_params.clear();
    }

    void Operator::clear_fields() {
        this->m_optional_fields.clear();
        for (auto &param : this->m_required_fields) {
            this->m_params.erase(param);
        }
        this->m_required_fields.clear();
    }

    void Operator::field(const std::string &param, Operator::FieldAttr attr, const Tensor &default_value) {
        this->field(param, attr);
        m_params.insert(std::make_pair(param, default_value));
    }

    void Operator::field(const std::string &param, Operator::FieldAttr attr) {
        switch (attr) {
            default:
                break;
            case OPTIONAL:
                this->m_optional_fields.insert(param);
                break;
            case REQUIRED:
                this->m_required_fields.insert(param);
                this->m_params.erase(param);
                break;
        }
    }

    std::vector<std::string> Operator::unsatisfied_fields() const {
        std::vector<std::string> fileds;
        for (auto &param : this->m_required_fields) {
            auto param_it = this->m_params.find(param);
            if (param_it == this->m_params.end() || param_it->second.empty()) {
                fileds.push_back(param);
            }
        }
        return fileds;
    }

    bool Operator::check_params() const {
        for (auto &param : this->m_required_fields) {
            auto param_it = this->m_params.find(param);
            if (param_it == this->m_params.end() || param_it->second.empty()) {
                return false;
            }
        }
        return true;
    }

    std::string Operator::fuzzy_field_name(const std::string &name) {
        if (m_required_fields.empty() || m_optional_fields.empty()) return "";
        int min_edit_distance = INT_MAX;
        std::string closest_name;
        for (auto &target_name : m_required_fields) {
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        for (auto &target_name : m_optional_fields) {
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        return closest_name;
    }

    std::string Operator::fuzzy_param_name(const std::string &name) {
        if (m_params.empty()) return "";
        int min_edit_distance = INT_MAX;
        std::string closest_name;
        for (auto &param_tensor_pair : m_params) {
            auto &target_name = param_tensor_pair.first;
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        return closest_name;
    }

    bool Operator::is_in_fields(const std::string &name) {
        return m_optional_fields.find(name) != m_optional_fields.end() ||
               m_required_fields.find(name) != m_required_fields.end();
    }

    bool Operator::is_in_params(const std::string &name) {
        return m_params.find(name) != m_params.end();
    }
}