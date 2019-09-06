//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_TENSOR_H
#define TENSORSTACK_THIRD_DRAGON_TENSOR_H

#include <core/tensor.h>

#include <numeric>

#include "type_meta.h"

namespace ts {
    namespace dragon {

        class Tensor {
        public:
            Tensor() = default;

            Tensor(const ts::Tensor &tst) : m_tst(tst) {}

            operator ts::Tensor() const { return m_tst; }

            int64_t dim(int64_t i) const { return int64_t(m_tst.size(int(i))); }

            Tensor *Reshape(DTYPE dtype, const std::vector<int64_t> &shape) {
                auto ts_shape = std::vector<int>(shape.begin(), shape.end());
                auto count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                if (m_tst.dtype() == dtype && m_tst.count() == count) {
                    m_tst.reshape(ts_shape);
                    return this;
                }
                m_tst = ts::Tensor();
                m_tst = ts::Tensor(ts::Tensor::InFlow::DEVICE, dtype, ts_shape);
                return this;
            }

            template<typename T>
            Tensor *Reshape(const std::vector<int64_t> &shape) {
                return Reshape(dtypeid<T>::id, shape);
            }

            Tensor *ReshapeLike(const Tensor &other) {
                auto &shape = other.m_tst.sizes();
                return Reshape(other.m_tst.dtype(), std::vector<int64_t>(shape.begin(), shape.end()));
            }

            template<typename T>
            bool IsType() const { return m_tst.dtype() == dtypeid<T>::id; }

            int64_t count() const {
                return int64_t(m_tst.count());
            }

            template<typename T, typename Context>
            T *mutable_data() {
                if (!IsType<T>()) {
                    TS_LOG_ERROR << "Expected dtype = " << type_str(dtypeid<T>::id)
                                 << " got " << type_str(m_tst.dtype()) << eject;
                }
                return reinterpret_cast<T *>(this->mutable_data_ptr<Context>());
            }

            template<typename T, typename Context>
            const T *data() const {
                if (!IsType<T>()) {
                    TS_LOG_ERROR << "Expected dtype = " << type_str(dtypeid<T>::id)
                                 << " got " << type_str(m_tst.dtype()) << eject;
                }
                return reinterpret_cast<const T *>(this->const_data_ptr<Context>());
            }

            template<typename Context>
            void *mutable_data_ptr() {
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    m_tst = m_tst.view(ts::Tensor::InFlow::HOST);
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    m_tst = m_tst.view(ts::Tensor::InFlow::DEVICE);
                }
                m_tst.broadcast();
                return m_tst.data();
            }

            template<typename Context>
            const void *const_data_ptr() const {
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    m_tst = m_tst.view(ts::Tensor::InFlow::HOST);
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    m_tst = m_tst.view(ts::Tensor::InFlow::DEVICE);
                }
                return m_tst.data();
            }

            std::string name() const { return ""; }

            ts::Tensor::Prototype meta() const { return m_tst.proto(); }

        private:
            mutable ts::Tensor m_tst;
        };
    }
}

#define XIsType(x, T) ((x).template IsType<T>())

#endif //TENSORSTACK_THIRD_DRAGON_TENSOR_H
