//
// Created by kier on 2019/3/4.
//

#ifndef TENSORSTACK_TEST_CASE_HPP
#define TENSORSTACK_TEST_CASE_HPP

#include <string>
#include <map>
#include <regex>
#include <fstream>
#include <algorithm>
#include <cmath>

#include <core/tensor.h>
#include <module/io/fstream.h>
#include <global/operator_factory.h>
#include <module/bubble.h>
#include <core/tensor_builder.h>
#include <runtime/stack.h>
#include <core/device_context.h>
#include <utils/ctxmgr_lite.h>
#include <runtime/workbench.h>

namespace ts {
    static inline std::string plot_line(const std::vector<Tensor> &input, const std::vector<Tensor> &output) {
        std::ostringstream oss;
        oss << "{";
        for (size_t i = 0; i < input.size(); ++i) {
            auto &tensor = input[i];
            if (i) oss << ", ";
            oss << type_str(tensor.dtype()) << ":" << to_string(tensor.sizes());
        }
        oss << "} -> {";
        for (size_t i = 0; i < output.size(); ++i) {
            auto &tensor = output[i];
            if (i) oss << ", ";
            oss << type_str(tensor.dtype()) << ":" << to_string(tensor.sizes());
        }
        oss << "}";
        return oss.str();
    }

    static inline std::string plot_line(const std::vector<Tensor> &input, const std::vector<Tensor::Prototype> &output) {
        std::ostringstream oss;
        oss << "{";
        for (size_t i = 0; i < input.size(); ++i) {
            auto &tensor = input[i];
            if (i) oss << ", ";
            oss << type_str(tensor.dtype()) << ":" << to_string(tensor.sizes());
        }
        oss << "} -> {";
        for (size_t i = 0; i < output.size(); ++i) {
            auto &tensor = output[i];
            if (i) oss << ", ";
            oss << type_str(tensor.dtype()) << ":" << to_string(tensor.sizes());
        }
        oss << "}";
        return oss.str();
    }

    static inline bool check_output(const std::vector<Tensor> &input, const std::vector<Tensor> &output) {
        if (input.size() != output.size()) return false;
        for (size_t i = 0; i < input.size(); ++i) {
            auto &a = input[i];
            auto &b = output[i];
            if (a.dtype() != b.dtype()) return false;
            if (!a.has_shape(b.sizes())) return false;
        }
        return true;
    }

    static inline bool check_output(const std::vector<Tensor> &input, const std::vector<Tensor::Prototype> &output) {
        if (input.size() != output.size()) return false;
        for (size_t i = 0; i < input.size(); ++i) {
            auto &a = input[i];
            auto &b = output[i];
            if (a.dtype() != b.dtype()) return false;
            if (!a.has_shape(b.sizes())) return false;
        }
        return true;
    }

    static inline void diff(const Tensor &x, const Tensor &y, float &max, float &avg) {
        auto float_x = tensor::cast(FLOAT32, x);
        auto float_y = tensor::cast(FLOAT32, y);
        auto count = std::min(x.count(), y.count());
        auto float_x_data = float_x.data<float>();
        auto float_y_data = float_y.data<float>();
        float local_max = 0;
        float local_sum = 0;
        for (int i = 0; i < count; ++i) {
            auto diff = std::fabs(float_x_data[i] - float_y_data[i]);
            local_sum += diff;
            if (local_max < diff) local_max = diff;
        }
        max = local_max;
        avg = local_sum / count;
    }

    class TestCase {
    public:
        std::string op;
        std::string name;
        int output_count;
        int param_count;
        int input_count;
        std::map<std::string, Tensor> param;
        std::map<int, Tensor> input;
        std::map<int, Tensor> output;

        // try load test case in files, throw exception if there is an broken case
        bool load(const std::string &root, const std::vector<std::string> &filenames) {
            TestCase tc;
            for (auto &filename : filenames) {
                if (filename.length() < 5 || filename[1] != '.') continue;

                auto fullpath = (root.empty() ? std::string() : root + "/");
                fullpath += filename;

                auto type_str = filename.substr(0, 1);
                auto type = std::strtol(type_str.c_str(), nullptr, 10);
                switch (type) {
                    default:
                        continue;
                    case 0: {
                        std::regex pattern(R"(^0\.(.*)\.txt$)");
                        std::smatch matched;
                        if (!std::regex_match(filename, matched, pattern)) continue;
                        std::fstream ifile(fullpath);
                        if (!(ifile >> tc.param_count >> tc.input_count >> tc.output_count)) {
                            TS_LOG_ERROR << "format error in: " << fullpath << eject;
                            return false;
                        }
                        if (!tc.op.empty()) {
                            TS_LOG_ERROR << "Found two operator description in " << root << ": " << matched.str(1) << " vs. " << tc.op << eject;
                        }
                        tc.op = matched.str(1);
                        break;
                    }
                    case 1: {
                        std::regex pattern(R"(^1\.(.*)\.t$)");
                        std::smatch matched;
                        if (!std::regex_match(filename, matched, pattern)) continue;
                        FileStreamReader ifile(fullpath);
                        std::string name = matched.str(1);
                        Tensor value;
                        value.externalize(ifile);
                        tc.param.insert(std::make_pair(std::move(name), std::move(value)));
                        break;
                    }
                    case 2: {
                        std::regex pattern(R"(^2\.input_(.*)\.t$)");
                        std::smatch matched;
                        if (!std::regex_match(filename, matched, pattern)) continue;
                        FileStreamReader ifile(fullpath);
                        std::string id_str = matched.str(1);
                        auto id = int(std::strtol(id_str.c_str(), nullptr, 10));
                        Tensor value;
                        value.externalize(ifile);
                        tc.input.insert(std::make_pair(id, std::move(value)));
                        break;
                    }
                    case 3: {
                        std::regex pattern(R"(^3\.output_(.*)\.t$)");
                        std::smatch matched;
                        if (!std::regex_match(filename, matched, pattern)) continue;
                        FileStreamReader ifile(fullpath);
                        std::string id_str = matched.str(1);
                        auto id = int(std::strtol(id_str.c_str(), nullptr, 10));
                        Tensor value;
                        value.externalize(ifile);
                        tc.output.insert(std::make_pair(id, std::move(value)));
                        break;
                    }
                }
            }
            // not an test case
            if (tc.op.empty()) return false;

            // check format
            if (tc.param.size() != tc.param_count) {
                TS_LOG_ERROR << "Param count mismatch in " << root << ": "
                             << tc.param_count << " needed with " << tc.param.size() << " given." << eject;
            }
            if (tc.input.size() != tc.input_count) {
                TS_LOG_ERROR << "Input count mismatch in " << root << ": "
                             << tc.input_count << " needed with " << tc.input.size() << " given." << eject;
            }
            for (int i = 0; i < tc.input_count; ++i) {
                if (tc.input.find(i) == tc.input.end()) {
                    TS_LOG_ERROR << "Input missing in " << root << ": "
                                 << "1.input_" << i << ".t needed." << eject;
                }
            }
            if (tc.output.size() != tc.output_count) {
                TS_LOG_ERROR << "Output count mismatch in " << root << ": "
                             << tc.output_count << " needed with " << tc.output.size() << " given." << eject;
            }
            for (int i = 0; i < tc.output_count; ++i) {
                if (tc.output.find(i) == tc.output.end()) {
                    TS_LOG_ERROR << "Output missing in " << root << ": "
                                 << "1.output_" << i << ".t needed." << eject;
                }
            }

            *this = std::move(tc);

            // format succeed
            return true;
        }

        bool run(const ComputingDevice &device, int loop_count = 100) {
            m_log.str("");

            if (op.empty()) {
                m_log << "[ERROR]: " << "operator is empty." << std::endl;
                return false;
            }

            Workbench bench(device);

            Bubble bubble(op, op, output_count);
            for (auto &param_pair: param) {
                bubble.set(param_pair.first, param_pair.second);
            }

            auto built_op = bench.offline_create(bubble, true);

            if (built_op == nullptr) {
                m_log << "[ERROR]: " << "Not supported operator \"" << op << "\" for " << device << std::endl;
                return false;
            }

            std::vector<Tensor> input_vector(input_count);
            std::vector<Tensor> output_vector(output_count);

            for (auto &input_pair : input) {
                input_vector[input_pair.first] = input_pair.second;
            }

            for (auto &output_pair : output) {
                output_vector[output_pair.first] = output_pair.second;
            }

            std::vector<Tensor::Prototype> output_protos;
            bench.offline_infer(built_op, input_vector, output_protos);

            m_log << "Wanted: " << plot_line(input_vector, output_vector) << std::endl;

            if (!check_output(output_vector, output_protos)) {
                m_log << "Infer:  " << plot_line(input_vector, output_protos) << std::endl;
                return false;
            }

            std::vector<Tensor> run_output;
            bench.offline_run(built_op, input_vector, run_output);

            if (!check_output(output_vector, run_output)) {
                m_log << "Run:    " << plot_line(input_vector, run_output) << std::endl;
                return false;
            }

            static const float MAX_MAX = 1e-4;
            static const float MAX_AVG = 1e-5;

            // check diff
            bool succeed = true;
            float max, avg;
            for (int i = 0; i < output_count; ++i) {
                auto &x = output_vector[i];
                auto &y = run_output[i];
                diff(x, y, max, avg);
                if (max > MAX_MAX || avg > MAX_AVG)  {
                    m_log << "[FAILED] Diff output " << i << ": max = " << max << ", " << "avg = " << avg << std::endl;
                    succeed = false;
                } else {
                    m_log << "[OK] Diff output " << i << ": max = " << max << ", " << "avg = " << avg << std::endl;
                }
            }

            if (!succeed) return false;

            return true;
        }

        std::string log() { return m_log.str(); }

    private:
        std::ostringstream m_log;
    };
}

#endif //TENSORSTACK_TEST_CASE_HPP
