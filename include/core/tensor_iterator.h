//
// Created by kier on 2019/2/21.
//

#ifndef TENSORSTACK_CORE_TENSOR_ITERATOR_H
#define TENSORSTACK_CORE_TENSOR_ITERATOR_H

#include <vector>
#include "utils/except.h"

namespace ts {
    class ShapeIterator {
    public:
        using self = ShapeIterator;

        explicit ShapeIterator(const std::vector<int> &shape)
                : m_shape(shape) {
            m_coordinate.resize(m_shape.size(), 0);
        }

        ShapeIterator(const std::vector<int> &shape, const std::vector<int> &coordinate)
                : m_shape(shape), m_coordinate(coordinate) {
        }

        self &operator++() {
            increase();
            return *this;
        }

        const self operator++(int) {
            auto temp = self(m_shape, m_coordinate);
            increase();
            return temp;
        }

        self &operator--() {
            decrease();
            return *this;
        }

        const self operator--(int) {
            auto temp = self(m_shape, m_coordinate);
            decrease();
            return temp;
        }

        void rewind() {
            for (auto &i : m_coordinate) {
                i = 0;
            }
        }

        const std::vector<int> &coordinate() {
            return m_coordinate;
        }

    private:
        void increase() {
            increase(int(m_coordinate.size() - 1));
        }

        void decrease() {
            decrease(int(m_coordinate.size() - 1));
        }

        void increase(int dim) {
            while (dim >= 0) {
                if (m_coordinate[dim] + 1 == m_shape[dim]) {
                    m_coordinate[dim] = 0;
                    --dim;
                    continue;
                } else {
                    m_coordinate[dim]++;
                    break;
                }
            }
        }

        void decrease(int dim) {
            while (dim >= 0) {
                if (m_coordinate[dim] - 1 < 0) {
                    m_coordinate[dim] = m_shape[dim] - 1;
                    --dim;
                    continue;
                } else {
                    m_coordinate[dim]--;
                    break;
                }
            }
        }


        std::vector<int> m_shape;
        std::vector<int> m_coordinate;

    public:
        ShapeIterator(const self &other) = default;
        ShapeIterator &operator=(const self &other) = default;

        ShapeIterator(self &&other) {
            *this = std::move(other);
        }
        ShapeIterator &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_shape);
            MOVE_MEMBER(m_coordinate);
#undef MOVE_MEMBER
            return *this;
        }
    };

    class HypeShape {
    public:
        using self = HypeShape;
        using T = int;

        explicit HypeShape(const std::vector<int> &shape)
                : m_shape(shape) {
            // update weights
            if (m_shape.empty()) return;
            m_weights.resize(m_shape.size());
            auto size = m_shape.size();
            auto weight_it = m_weights.rbegin();
            auto shape_it = m_shape.rbegin();
            *weight_it++ = *shape_it++;
            for (size_t times = size - 1; times; --times) {
                *weight_it = *(weight_it - 1) * *shape_it;
                ++weight_it;
                ++shape_it;
            }
        }

        T to_index(const std::initializer_list<T> &coordinate) const {
            // if (coordinate.size() > m_shape.size()) throw CoordinateOutOfShapeException(m_shape, coordinate);
            if (coordinate.size() == 0) return 0;
            auto size = coordinate.size();
            auto weight_it = m_weights.end() - size + 1;
            auto coordinate_it = coordinate.begin();
            T index = 0;
            for (size_t times = size - 1; times; --times) {
                index += *weight_it * *coordinate_it;
                ++weight_it;
                ++coordinate_it;
            }
            index += *coordinate_it;
            return index;
        }

        T to_index(const std::vector<T> &coordinate) const {
            // if (coordinate.size() > m_shape.size()) throw CoordinateOutOfShapeException(m_shape, coordinate);
            if (coordinate.empty()) return 0;
            auto size = coordinate.size();
            auto weight_it = m_weights.end() - size + 1;
            auto coordinate_it = coordinate.begin();
            T index = 0;
            for (size_t times = size - 1; times; --times) {
                index += *weight_it * *coordinate_it;
                ++weight_it;
                ++coordinate_it;
            }
            index += *coordinate_it;
            return index;
        }

        std::vector<T> to_coordinate(T index) const {
            // if (m_shape.empty()) return std::vector<T>();
            // if (index >= m_weights[0]) throw IndexOutOfShapeException(m_shape, index);
            if (m_shape.empty())
                return std::vector<T>();
            std::vector<T> coordinate(m_shape.size());
            to_coordinate(index, coordinate);
            return std::move(coordinate);
        }

        void to_coordinate(T index, std::vector<T> &coordinate) const {
            if (m_shape.empty()) {
                coordinate.clear();
                return;
            }
            coordinate.resize(m_shape.size());
            auto size = m_shape.size();
            auto weight_it = m_weights.begin() + 1;
            auto coordinate_it = coordinate.begin();
            for (size_t times = size - 1; times; --times) {
                *coordinate_it = index / *weight_it;
                index %= *weight_it;
                ++weight_it;
                ++coordinate_it;
            }
            *coordinate_it = index;
        }

        T count() const { return m_weights.empty() ? 1 : m_weights[0]; }

        T weight(size_t i) const { return m_weights[i]; };

        const std::vector<T> &weight() const { return m_weights; };

        T shape(size_t i) const { return m_shape[i]; };

        const std::vector<T> &shape() const { return m_shape; };

        explicit operator std::vector<int>() const { return m_shape; }

    private:
        std::vector<int> m_shape;
        std::vector<T> m_weights;

    public:
        HypeShape(const self &other) = default;
        HypeShape &operator=(const self &other) = default;

        HypeShape(self &&other) {
            *this = std::move(other);
        }
        HypeShape &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_shape);
            MOVE_MEMBER(m_weights);
#undef MOVE_MEMBER
            return *this;
        }
    };
}

#endif //TENSORSTACK_TENSOR_ITERATOR_H
