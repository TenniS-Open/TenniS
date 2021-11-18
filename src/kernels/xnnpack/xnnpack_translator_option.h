//
// Created by sen on 2021/9/27.
//

#ifndef TENNIS_XNNPACK_TRANSLATOR_OPTION_H
#define TENNIS_XNNPACK_TRANSLATOR_OPTION_H

#include "compiler/option/translator_option.h"

namespace ts {

    class XnnpackTranslatorOption : public TranslatorOption {
    public:
        bool translate(const ComputingDevice &device,
                       const Node node,
                       Node &translated_node,
                       const std::string &params,
                       bool output_flag) const final;
    };
}

#endif //TENNIS_XNNPACK_TRANSLATOR_OPTION_H
