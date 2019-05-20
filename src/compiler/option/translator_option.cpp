#include "compiler/option/translator_option.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "module/menu.h"

namespace ts {

    static std::map<const TranslatorOption*, std::set<const std::string>> &GetStaticTranslateOptions() {
        static std::map<const TranslatorOption*, std::set<const std::string>> options;
        return options;
    }

    const std::map<const TranslatorOption*, std::set<const std::string>> &GetFullTranslateOptions() {
        return GetStaticTranslateOptions();
    }

    void RegisterTranslateOption(const TranslatorOption *option, std::initializer_list<const std::string>& op_list) {
        auto &options = GetStaticTranslateOptions();
        auto ops = std::set<const std::string>(op_list.begin(), op_list.end());
        options.insert(std::make_pair(option, ops));
    }

    static std::map<const TranslatorOption*, std::set<const std::string>> &MapOptionSupportOperator() {
        static std::map<const TranslatorOption*, std::set<const std::string>> map_option_support_operator;
        return map_option_support_operator;
    }

    class Fp16TranslatorOption : public TranslatorOption {


        bool translate(const ComputingDevice &device, 
                       Node node, 
                       Node &zipped_node, 
                       std::set<const std::string> operators,
                       bool output_flag) const final {
            //NOTE: only support gpu device now
            if (device.type() != GPU)
                return false;

            //check if support current op
            if (operators.find(node.bubble().op()) == operators.end()) {
                return false;
            }

            auto inputs = node.inputs();
            std::vector<Node> cast_fp16_nodes;
            for ( auto& input : inputs )
            {
                auto cast_fp16_node = bubble::op(node.bubble().name() + "_cast_fp16", name::layer::cast(), {input});

                //cast_fp16_node.bubble().set(name::dtype, ts::FLOAT16);
                cast_fp16_nodes.emplace_back(cast_fp16_nodes);
            }

            zipped_node = bubble::op(node.bubble().name(), node.bubble().op(), { cast_fp16_nodes });

            return true;
        }
    };
}

//regist translate option
//TS_REGISTER_TRANSLATOR_OPTION(ts::Fp16TranslatorOption, { ts::name::layer::conv2d() });

//regist operators that translator's option support
