//
// Created by seeta on 2018/5/19.
//

#ifndef TENSORSTACK_RUNTIME_OPERATOR_H
#define TENSORSTACK_RUNTIME_OPERATOR_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <core/tensor.h>

namespace ts {
    class Stack;

    class Operator {
    public:
        using self = Operator;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        template<typename K, typename V>
        using hash_map = std::unordered_map<K, V>;

        template<typename K>
        using hash_set = std::unordered_set<K>;

        enum FieldAttr {
            OPTIONAL,
            REQUIRED,
        };

        virtual ~Operator() = default;

        // explicit Operator(const std::string &name) : m_name(name) {}
        virtual void init();

        // work on tensor stack, same as call
        virtual int run(Stack &stack) = 0;

        // infer output size by input size, only for size allocate
        virtual int infer(Stack &stack, std::vector<Tensor::Prototype> &output) = 0;

        bool has(const std::string &param) const;

        void set(const std::string &param, const Tensor &value);

        Tensor &get(const std::string &param);

        const Tensor &get(const std::string &param) const;

        void clear(const std::string &param);

        void clear_params();

        /**
         * check if all required fields have been set
         * @return return true only if all required fields set
         */
        bool check_params() const;

        std::vector<std::string> unsatisfied_fields() const;

        void field(const std::string &param, FieldAttr attr, const Tensor &default_value);

        void field(const std::string &param, FieldAttr attr);

        void clear_fields();

    private:
        bool is_in_fields(const std::string &name);

        bool is_in_params(const std::string &name);

        std::string fuzzy_field_name(const std::string &name);

        std::string fuzzy_param_name(const std::string &name);

        hash_map<std::string, Tensor> m_params;
        hash_set<std::string> m_optional_fields;
        hash_set<std::string> m_required_fields;
        // std::string m_name;
    };
}


#endif //TENSORSTACK_RUNTIME_OPERATOR_H
