//
// Created by kier on 2018/10/30.
//

#include <cassert>
#include <cstring>
#include <unordered_set>
#include "core/tensor_converter.h"

namespace ts {
    namespace tensor {
        Tensor from(const std::string &value) {
            auto length = value.size();
            Tensor tensor(CHAR8, Shape({int(length)}));
            std::memcpy(tensor.data(), value.data(), length);
            return tensor;
        }

        std::string to_string(const Tensor &value) {
            assert(value.proto().dtype() == CHAR8);
            assert(value.proto().sizes().size() == 1);
            auto cpu_value = value;
            if (cpu_value.device().type() != CPU) {
                auto controller = std::make_shared<DynamicMemoryController>(MemoryDevice(CPU));
                cpu_value = cpu_value.clone(controller);
            }
            auto length = cpu_value.proto().sizes()[0];
            return std::string(cpu_value.data<char>(), size_t(length));
        }

        template <DTYPE TYPE_DST, DTYPE TYPE_SRC>
        static void type_cast(typename dtype<TYPE_DST>::declare *dst, const typename dtype<TYPE_SRC>::declare *src, size_t size) {
            size_t i = 0;
            for (; i + 4 <= size; i += 4) {
                *dst = static_cast<typename dtype<TYPE_DST>::declare>(*src); ++dst; ++src;
                *dst = static_cast<typename dtype<TYPE_DST>::declare>(*src); ++dst; ++src;
                *dst = static_cast<typename dtype<TYPE_DST>::declare>(*src); ++dst; ++src;
                *dst = static_cast<typename dtype<TYPE_DST>::declare>(*src); ++dst; ++src;
            }
            for (; i < size; ++i) {
                *dst = static_cast<typename dtype<TYPE_DST>::declare>(*src); ++dst; ++src;
            }
        };

        template <DTYPE TYPE_SRC>
        static void type_cast_to(void *dst, DTYPE dst_type, const typename dtype<TYPE_SRC>::declare *src, size_t size) {
            switch (dst_type)
            {
                default: throw Exception( std::string("Can not convert dtype ") + type_str(TYPE_SRC) + " to " + type_str(dst_type));
#define __CASE_TYPE_CALL_TYPE_CAST(__type__) \
                case __type__: type_cast<__type__, TYPE_SRC>(reinterpret_cast<typename dtype<__type__>::declare *>(dst), src, size); break;
                __CASE_TYPE_CALL_TYPE_CAST(INT8)
                __CASE_TYPE_CALL_TYPE_CAST(UINT8)
                __CASE_TYPE_CALL_TYPE_CAST(INT16)
                __CASE_TYPE_CALL_TYPE_CAST(UINT16)
                __CASE_TYPE_CALL_TYPE_CAST(INT32)
                __CASE_TYPE_CALL_TYPE_CAST(UINT32)
                __CASE_TYPE_CALL_TYPE_CAST(INT64)
                __CASE_TYPE_CALL_TYPE_CAST(UINT64)
                __CASE_TYPE_CALL_TYPE_CAST(FLOAT16)
                __CASE_TYPE_CALL_TYPE_CAST(FLOAT32)
                __CASE_TYPE_CALL_TYPE_CAST(FLOAT64)
                __CASE_TYPE_CALL_TYPE_CAST(CHAR8)
                __CASE_TYPE_CALL_TYPE_CAST(CHAR16)
                __CASE_TYPE_CALL_TYPE_CAST(CHAR32)
#undef __CASE_TYPE_CALL_TYPE_CAST
            }
        }

        static void type_cast_to_from(void *dst, DTYPE dst_type, const void *src, DTYPE src_type, size_t size) {
            switch (src_type)
            {
                default: throw Exception( std::string("Can not convert dtype ") + type_str(src_type) + " to " + type_str(dst_type));
#define __CASE_TYPE_CALL_TYPE_CAST_TO(__type__) \
                case __type__: type_cast_to<__type__>(dst, dst_type, reinterpret_cast<const typename dtype<__type__>::declare *>(src), size); break;
                __CASE_TYPE_CALL_TYPE_CAST_TO(INT8)
                __CASE_TYPE_CALL_TYPE_CAST_TO(UINT8)
                __CASE_TYPE_CALL_TYPE_CAST_TO(INT16)
                __CASE_TYPE_CALL_TYPE_CAST_TO(UINT16)
                __CASE_TYPE_CALL_TYPE_CAST_TO(INT32)
                __CASE_TYPE_CALL_TYPE_CAST_TO(UINT32)
                __CASE_TYPE_CALL_TYPE_CAST_TO(INT64)
                __CASE_TYPE_CALL_TYPE_CAST_TO(UINT64)
                __CASE_TYPE_CALL_TYPE_CAST_TO(FLOAT16)
                __CASE_TYPE_CALL_TYPE_CAST_TO(FLOAT32)
                __CASE_TYPE_CALL_TYPE_CAST_TO(FLOAT64)
                __CASE_TYPE_CALL_TYPE_CAST_TO(CHAR8)
                __CASE_TYPE_CALL_TYPE_CAST_TO(CHAR16)
                __CASE_TYPE_CALL_TYPE_CAST_TO(CHAR32)
#undef __CASE_TYPE_CALL_TYPE_CAST_TO
            }
        }

        Tensor cast(DTYPE dtype, const Tensor &value) {
            auto cpu_value = value;
            if (cpu_value.device().type() != CPU) {
                auto controller = std::make_shared<DynamicMemoryController>(MemoryDevice(CPU));
                cpu_value = cpu_value.clone(controller);
            }
            auto controller = std::make_shared<DynamicMemoryController>(MemoryDevice(CPU));
            Tensor casted(controller, dtype, cpu_value.sizes());

            std::unordered_set<DTYPE> unsupported_types =
                    {UNKNOWN8, UNKNOWN16, UNKNOWN32, UNKNOWN64, UNKNOWN128, VOID, PTR};

            if (unsupported_types.find(dtype) != unsupported_types.end()
                || unsupported_types.find(cpu_value.dtype()) != unsupported_types.end()) {
                throw Exception(
                        std::string("Can not convert dtype ") + type_str(cpu_value.dtype()) + " to " + type_str(dtype));
            }

            type_cast_to_from(casted.data(), dtype, cpu_value.data(), cpu_value.dtype(), size_t(cpu_value.count()));

            return casted;
        }
    }

    template<typename T>
    Tensor tensor_builder<T>::build(const std::vector<T> &value) {
        auto controller = std::make_shared<DynamicMemoryController>(MemoryDevice(CPU));
        Tensor t(controller, dtypeid<T>::id, {int(value.size())});
        std::memcpy(t.data(), value.data(), value.size() * sizeof(T));
        return t;
    }
}

template class ts::tensor_builder<ts::dtype<ts::INT8>::declare>;
template class ts::tensor_builder<ts::dtype<ts::UINT8>::declare>;
template class ts::tensor_builder<ts::dtype<ts::INT16>::declare>;
template class ts::tensor_builder<ts::dtype<ts::UINT16>::declare>;
template class ts::tensor_builder<ts::dtype<ts::INT32>::declare>;
template class ts::tensor_builder<ts::dtype<ts::UINT32>::declare>;
template class ts::tensor_builder<ts::dtype<ts::INT64>::declare>;
template class ts::tensor_builder<ts::dtype<ts::UINT64>::declare>;
template class ts::tensor_builder<ts::dtype<ts::FLOAT32>::declare>;
template class ts::tensor_builder<ts::dtype<ts::FLOAT64>::declare>;
