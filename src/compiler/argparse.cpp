//
// Created by kier on 2019-06-10.
//

#include <compiler/argparse.h>

#include <regex>

#include <utils/log.h>

namespace ts {
    static std::string stand1(const std::string &str) {
        static std::regex regex(R"(^\s*(\S*)\s*$)");
        std::smatch match;
        std::string cvt;
        if(std::regex_match(str, match, regex)) {
            cvt = match[1];
        } else {
            cvt = str;
        }
        std::transform(cvt.begin(), cvt.end(), cvt.begin(), std::towlower);
        return cvt;
    }

    static std::string stand2(const std::string &str) {
        static std::regex regex(R"(^[_-]*([^_-]*)[_-]*$)");
        std::smatch match;
        if(std::regex_match(str, match, regex)) {
            return match[1];
        } else {
            return str;
        }
    }

    ArgParser::ArgParser() {
        m_map_device.insert(std::make_pair("", "cpu"));
        m_map_device.insert(std::make_pair("cuda", "gpu"));
    }

    void ts::ArgParser::add(const std::vector<std::string> &arg, const std::vector<std::string> &neg_arg,
                            bool default_value) {
        if (arg.empty()) {
            TS_LOG_ERROR << "param@1 can not be empty." << eject;
        }

        const std::string &true_arg_name = arg.front();
        m_true_arg_names.insert(std::make_pair(true_arg_name, true_arg_name));
        for (auto it = arg.begin() + 1; it != arg.end(); ++it) {
            m_true_arg_names.insert(std::make_pair(*it, true_arg_name));
        }
        for (auto it = neg_arg.begin(); it != neg_arg.end(); ++it) {
            m_false_arg_names.insert(std::make_pair(*it, true_arg_name));
        }
        if (default_value) {
            m_arg_value.insert(std::make_pair(true_arg_name, default_value));
        }
    }

    bool ArgParser::set(const std::string &arg) {
        auto true_it = m_true_arg_names.find(arg);
        if (true_it != m_true_arg_names.end()) {
            m_arg_value[true_it->second] = true;
            return true;
        }
        auto false_it = m_false_arg_names.find(arg);
        if (false_it != m_false_arg_names.end()) {
            // m_arg_value.erase(false_it->second);
            m_arg_value[false_it->second] = false;
            return true;
        }
        return false;
    }

    bool ArgParser::get(const std::string &arg) const {
        auto it = m_arg_value.find(arg);
        if (it != m_arg_value.end()) {
            return it->second;
        }
        return false;
    }

    void ArgParser::parse(const std::string &args) {
        static std::regex regex_map(R"(^-[vV]:([^:]*):([^:]*)$)");

        auto params = Split(args, " \t\r\n");
        std::smatch match_v;
        for (auto &param : params) {
            if (param.empty()) continue;
            // check if it's device map
            if(std::regex_match(param, match_v, regex_map)) {
                this->m_map_device[stand1(match_v[1])] = stand1(match_v[2]);
                continue;
            }
            // is param setting
            set(param);
        }
    }

    std::string ArgParser::map_device(const std::string &device) {
        auto stand_device = stand1(device);
        while (true) {
            auto it = this->m_map_device.find(stand_device);
            if (it != this->m_map_device.end()) {
                stand_device = it->second;
            } else {
                return stand2(stand_device);
            }
        }
    }
}
