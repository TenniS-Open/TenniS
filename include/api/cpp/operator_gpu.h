//
// Created by kier on 19-5-18.
//

#ifndef TENSORSTACK_API_CPP_OPERATOR_GPU_H
#define TENSORSTACK_API_CPP_OPERATOR_GPU_H

#include "operator.h"
#include "../operator_gpu.h"

namespace ts {
    namespace api {
        class OperatorGPU {
        public:
            virtual ~OperatorGPU() = default;

            virtual void init(const OperatorParams &params, cudaStream_t stream) = 0;

            virtual std::vector<std::pair<DTYPE, Shape>> infer(const std::vector<Tensor> &args, cudaStream_t stream) = 0;

            virtual std::vector<Tensor> run(const std::vector<Tensor> &args, cudaStream_t stream) = 0;

            static void Throw(const std::string &message) {
                ts_Operator_Throw(message.c_str());
            }

            static void Throw(const std::string &message, const std::string &filename, int32_t line_number) {
                ts_Operator_ThrowV2(message.c_str(), filename.c_str(), line_number);
            }

            static void Init(void *op, const ts_OperatorParams *params, void *stream) {
                auto obj = reinterpret_cast<OperatorGPU *>(op);
                obj->init(OperatorParams(params), TS_CUDA_STREAM(stream));
            }

            static ts_Tensor *Infer(void *op, int32_t argc, ts_Tensor **argv, void *stream) {
                auto obj = reinterpret_cast<OperatorGPU *>(op);
                auto proto = obj->infer(borrowed_tensors(argc, argv), TS_CUDA_STREAM(stream));
                if (proto.empty()) return nullptr;
                std::vector<int32_t> data;
                data.emplace_back(int32_t(proto.size()));
                for (auto &pair : proto) {
                    data.emplace_back(int32_t(pair.first));
                    data.emplace_back(int32_t(pair.second.size()));
                    for (auto dim : pair.second) {
                        data.emplace_back(dim);
                    }
                }
                auto count = int32_t(data.size());
                auto packed_tensor = ts_new_Tensor(&count, 1, TS_INT32, data.data());
                // TS_API_AUTO_CHECK(packed_tensor != nullptr); // check in tensorstack side
                return packed_tensor;
            }

            static ts_Tensor *Run(void *op, int32_t argc, ts_Tensor **argv, void *stream) {
                auto obj = reinterpret_cast<OperatorGPU *>(op);
                auto fields = obj->run(borrowed_tensors(argc, argv), TS_CUDA_STREAM(stream));
                if (fields.empty()) return nullptr;
                std::vector<ts_Tensor *> cfields;
                for (auto &field : fields) {
                    cfields.emplace_back(field.get_raw());
                }
                auto packed_raw = ts_Tensor_pack(cfields.data(), int32_t(cfields.size()));
                // TS_API_AUTO_CHECK(packed_raw != nullptr);    // check in tensorstack side
                return packed_raw;
            }
        };

        /**
         *
         * @tparam T must be the subclass of Operator
         * @param device register device
         */
        template<typename T>
        inline void RegisterOperatorGPU(const std::string &op) {
            ts_OperatorGPU_Register(op.c_str(),
                                    &NewAndFree<T>::New, &NewAndFree<T>::Free,
                                    &T::Init, &T::Infer, &T::Run);
        }
    }
}

#endif //TENSORSTACK_API_CPP_OPERATOR_GPU_H
