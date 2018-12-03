//
// Created by kier on 2018/12/3.
//

#ifndef TENSORSTACK_MODULE_HEADER_H
#define TENSORSTACK_MODULE_HEADER_H

#include "serialization.h"

#include <array>

#define TS_MODULE_CODE_V1 0x19910929

namespace ts {
    class Header : public Serializable {
    public:
        uint32_t fake = 0;
        uint32_t code = 0;
        std::array<char, 120> data = {0};

        size_t serialize(StreamWriter &stream) const final;

        size_t externalize(StreamReader &stream) final;
    };
}

#endif //TENSORSTACK_MODULE_HEADER_H