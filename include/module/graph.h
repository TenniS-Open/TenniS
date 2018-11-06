//
// Created by kier on 2018/10/15.
//

#ifndef TENSORSTACK_MODULE_GRAPH_H
#define TENSORSTACK_MODULE_GRAPH_H

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "utils/except.h"
#include "utils/assert.h"

namespace ts {

    /**
     * Tree node,
     */
    class _RawNode {
    public:
        using self = _RawNode;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using weak = std::weak_ptr<self>;  ///< smart pointer

        _RawNode() = default;

        virtual ~_RawNode() = default;

        const std::vector<weak> &inputs() const { return m_inputs; }

        const std::vector<weak> &outputs() const { return m_outputs; }

        template<typename T>
        T *ptr();

        template<typename T>
        const T *ptr() const { return const_cast<self *>(this)->ptr<T>(); }

        static void Link(const weak &node, const std::vector<weak> &inputs) {
            auto output = node.lock();
            if (!output) throw NullPointerException("Link expired node");
            output->m_inputs.resize(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto input = inputs[i].lock();
                if (!input) throw NullPointerException("Link expired node");
                input->m_outputs.push_back(output);
                output->m_inputs[i] = input;
            }
        }

        virtual std::string str() const {
            std::ostringstream oss;
            oss << "<Node: " << this << ">";
            return oss.str();
        }

        virtual std::string repr() const { return this->str(); }

    private:
        std::vector<weak> m_inputs;
        std::vector<weak> m_outputs;
    };

    template<typename T>
    class _RawNodeWithValue : public _RawNode {
    public:
        using self = _RawNodeWithValue;
        using supper = _RawNode;

        using _RawNode::_RawNode;

        template<typename... Args>
        explicit _RawNodeWithValue(Args &&...args)
                : m_value(std::forward<Args>(args)...) {}

        T &value() { return m_value; }

        const T &value() const { return m_value; }

        std::string str() const override {
            std::ostringstream oss;
            oss << "<Node: " << this->value() << ">";
            return oss.str();
        }

    private:
        T m_value;
    };

    template<typename T>
    T *_RawNode::ptr() {
        auto value_ptr = dynamic_cast<_RawNodeWithValue<T> *>(this);
        if (value_ptr == nullptr) return nullptr;
        return &value_ptr->value();
    }

    /**
     * Node only support single output
     * Use Pack node support multi output, like:
     *     c = func1(a, b) # c is pack node
     *     c:1 = unpack(c, 1)   # get c's 1st output
     *     c:2 = unpack(c, 2)   # get c's 2nd output
     *  Notice: The c is pack(c:1, c:2) node, and the unpack method's first parameter must be pack node
     *  TODO: supporting edit graph, not just a link
     */
    class Node {
    public:
        using self = Node;

        friend class Graph;

        Node(const self &) = default;

        Node(self &&) = default;

        Node &operator=(const self &) = default;

        Node &operator=(self &&) = default;

        std::vector<Node> inputs() const {
            auto ptr = m_ptr.lock();
            if (!ptr) throw NullPointerException("Getting expired node's inputs");
            auto raw_vector = ptr->inputs();
            std::vector<Node> out_vector;
            out_vector.reserve(raw_vector.size());
            for (auto &node : raw_vector) out_vector.emplace_back(Node(node));
            return std::move(out_vector);
        }

        std::vector<Node> outputs() const {
            auto ptr = m_ptr.lock();
            if (!ptr) throw NullPointerException("Getting expired node's outputs");
            auto raw_vector = ptr->outputs();
            std::vector<Node> out_vector;
            out_vector.reserve(raw_vector.size());
            for (auto &node : raw_vector) out_vector.emplace_back(Node(node));
            return std::move(out_vector);
        }

        void *ptr() const { return m_ptr.lock().get(); }

        template<typename T>
        T *ptr() {
            auto raw_ptr = m_ptr.lock();
            if (!raw_ptr) return nullptr;
            return raw_ptr->ptr<T>();
        }

        template<typename T>
        const T *ptr() const { return const_cast<self *>(this)->ptr<T>(); }

        template<typename T>
        T &ref() {
            auto value_ptr = this->ptr<T>();
            if (value_ptr == nullptr) throw NullPointerException("Getting reference from null pointer");
            return *value_ptr;
        }

        template<typename T>
        const T &ref() const { return const_cast<self *>(this)->ref<T>(); }

        static void Link(const Node &node, const std::vector<Node> &inputs) {
            std::vector<_RawNode::weak> raw_inputs;
            raw_inputs.reserve(inputs.size());
            for (auto &input : inputs) raw_inputs.emplace_back(_RawNode::weak(input));
            _RawNode::Link(node.m_ptr, raw_inputs);
        }

        std::string str() const {
            auto raw_ptr = m_ptr.lock();
            if (!raw_ptr) return "<Node: nil>";
            return raw_ptr->str();
        }

        std::string repr() const {
            auto raw_ptr = m_ptr.lock();
            if (!raw_ptr) return "<Node: nil>";
            return raw_ptr->repr();
        }

    private:
        explicit Node(const _RawNode::weak &ptr) : m_ptr(ptr) {}

        explicit operator _RawNode::weak() const { return m_ptr; }

        _RawNode::weak m_ptr;
    };

    inline std::ostream &operator<<(std::ostream &out, const Node &node) {
        return out << node.str();
    }

    /**
     * Graph, only saving nodes,
     * The Node generated by Graph will be disabled after destruction
     */
    class Graph {
    public:
        using self = Graph;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        template<typename T, typename... Args>
        Node make(Args &&...args) {
            auto node = std::make_shared<_RawNodeWithValue<T>>(std::forward<Args>(args)...);
            m_nodes.push_back(node);
            return Node(node);
        }

        std::vector<Node> nodes() const {
            std::vector<Node> out_vector;
            out_vector.reserve(m_nodes.size());
            for (auto &node : m_nodes) out_vector.emplace_back(Node(node));
            return std::move(out_vector);
        }

    private:
        std::vector<_RawNode::shared> m_nodes;
    };

    inline bool operator==(const Node &lhs, const Node &rhs) { return lhs.ptr() == rhs.ptr(); }

    inline bool operator!=(const Node &lhs, const Node &rhs) { return lhs.ptr() != rhs.ptr(); }

    inline bool operator<(const Node &lhs, const Node &rhs) { return lhs.ptr() < rhs.ptr(); }

    inline bool operator>(const Node &lhs, const Node &rhs) { return lhs.ptr() > rhs.ptr(); }

    inline bool operator<=(const Node &lhs, const Node &rhs) { return lhs.ptr() <= rhs.ptr(); }

    inline bool operator>=(const Node &lhs, const Node &rhs) { return lhs.ptr() >= rhs.ptr(); }
}

namespace std {
    template<>
    struct hash<ts::Node> {
        std::size_t operator()(const ts::Node &key) const {
            using std::size_t;
            using std::hash;

            return hash<void *>()(key.ptr());
        }
    };
}


#endif //TENSORSTACK_MODULE_GRAPH_H
