//
// Created by kier on 2020/6/3.
//

#ifndef TENNIS_MODULE_NODE_H
#define TENNIS_MODULE_NODE_H

#include "bubble.h"
#include <functional>

namespace ts {
    namespace v2 {
        template<typename T>
        class LinkedValue {
        public:
            using self = LinkedValue;
            using Value = T;
            using shared = std::shared_ptr<self>;
            using set = std::set<self *>;
            using vector = std::vector<shared>;

            LinkedValue() = default;

            template<typename... Args>
            explicit LinkedValue(Args &&...args)
                    : m_value(std::forward<Args>(args)...) {}

            LinkedValue(const self &) = default;

            LinkedValue &operator=(const self &) = default;

            LinkedValue(self &&) = default;

            LinkedValue &operator=(self &&) = default;

            void link(const std::vector<shared> &inputs) {
                set node_depends;
                for (auto &i : inputs) {
                    auto &i_depends = i->depends();
                    node_depends.insert(i_depends.begin(), i_depends.end());
                    node_depends.insert(i.get());
                }
                if (node_depends.find(this) != node_depends.end()) {
                    TS_LOG_ERROR << "Can not do loop link." << eject;
                }
                m_inputs = inputs;
                m_depends = node_depends;
            }

            const std::set<self*> &depends() const { return m_depends; }

            const std::vector<shared> &inputs() const { return m_inputs; }

            T &value() { return m_value; }

            const T &value() const { return m_value; }

            std::string str() const {
                std::ostringstream oss;
                oss << "<Node: " << this->value() << ">";
                return oss.str();
            }

            std::string repr() const { return this->str(); }

            T *ptr() { return &m_value; }

            const T *ptr() const { return &m_value; }

            T &ref() { return m_value; }

            const T &ref() const { return m_value; }

        private:
            T m_value;

            vector m_inputs;
            set m_depends;  // depends including inputs, no ref counts
        };

        using LinkedBubble = LinkedValue<Bubble>;

        class Node {
        public:
            using self = Node;

            Node()
                    : m_raw(std::make_shared<LinkedBubble>()) {
            }

            template<typename... Args>
            explicit Node(Args &&...args)
                    : m_raw(std::make_shared<LinkedBubble>(std::forward<Args>(args)...)) {
            }

            Node(const self &) = default;

            Node &operator=(const self &) = default;

            Node(self &&) = default;

            Node &operator=(self &&) = default;

            void link(const std::vector<Node> &inputs) {
                LinkedBubble::vector raw_inputs;
                raw_inputs.reserve(inputs.size());
                for (auto &i : inputs) {
                    raw_inputs.emplace_back(i.m_raw);
                }
                m_raw->link(raw_inputs);
            }

            static void Link(Node &node, const std::vector<Node> &inputs) {
                node.link(inputs);
            }

            static void Link(Node &node, const Node &input) {
                node.link({input});
            }

            std::vector<Node> inputs() const {
                auto &raw_vector = m_raw->inputs();
                std::vector<Node> out_vector;
                out_vector.reserve(raw_vector.size());
                for (auto &node : raw_vector) out_vector.emplace_back(Node(node));
                return std::move(out_vector);
            }

            Node input(int i) const { return inputs()[i]; }

            Node input(size_t i) const { return inputs()[i]; }

            std::string str() const {
                return m_raw->str();
            }

            std::string repr() const {
                return m_raw->repr();
            }

            Bubble &bubble() {
                return m_raw->value();
            }

            const Bubble &bubble() const {
                return m_raw->value();
            }

            Bubble *operator->() {
                return &bubble();
            }

            const Bubble *operator->() const {
                return &bubble();
            }

            Bubble &operator*() {
                return bubble();
            }

            const Bubble &operator*() const {
                return bubble();
            }

            void *ptr() { return m_raw.get(); }

            const void *ptr() const { return m_raw.get(); }

        private:
            LinkedBubble::shared m_raw;

            explicit Node(LinkedBubble::shared raw) : m_raw(std::move(raw)) {}
        };

        inline bool operator==(const Node &lhs, const Node &rhs) { return lhs.ptr() == rhs.ptr(); }

        inline bool operator!=(const Node &lhs, const Node &rhs) { return lhs.ptr() != rhs.ptr(); }

        inline bool operator<(const Node &lhs, const Node &rhs) { return lhs.ptr() < rhs.ptr(); }

        inline bool operator>(const Node &lhs, const Node &rhs) { return lhs.ptr() > rhs.ptr(); }

        inline bool operator<=(const Node &lhs, const Node &rhs) { return lhs.ptr() <= rhs.ptr(); }

        inline bool operator>=(const Node &lhs, const Node &rhs) { return lhs.ptr() >= rhs.ptr(); }
    }
    using namespace v2;
}

namespace std {
    template<>
    struct hash<ts::v2::Node> {
        std::size_t operator()(const ts::v2::Node &key) const {
            using std::size_t;
            using std::hash;

            return hash<const void *>()(key.ptr());
        }
    };
}

#endif //TENNIS_MODULE_NODE_H
