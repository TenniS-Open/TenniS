//
// Created by kier on 2019/1/28.
//

#ifndef TENSORSTACK_BOARD_BOARD_H
#define TENSORSTACK_BOARD_BOARD_H


#include <vector>
#include <list>
#include <unordered_map>
#include <numeric>

namespace ts {
    template <typename T>
    class Statistics {
    public:
        using self = Statistics;
        using Datum = T;

        void append(const Datum datum) {
            this->m_data.push_back(datum);
        }

        void clear() {
            this->m_data.clear();
        }

        Datum max() const {
            if (this->m_data.empty()) return Datum(0);
            return std::minmax(this->m_data.begin(), this->m_data.end()).second;
        }

        Datum min() const {
            if (this->m_data.empty()) return Datum(0);
            return std::minmax(this->m_data.begin(), this->m_data.end()).first;
        }

        Datum avg() const {
            return this->sum() / this->count();
        }

        Datum sum() const {
            return std::accumulate(this->m_data.begin(), this->m_data.end(), 0);
        }

        size_t count() const {
            return this->m_data.size();
        }

    private:
        std::list<Datum> m_data;
    };

    template <typename T>
    class Board {
    public:
        using self = Board;
        using Datum = typename Statistics<T>::Datum;
        using name_type = std::string;
        using value_type = Statistics<T>;
        using pair = std::pair<name_type, value_type>;

        void append(const name_type &name, const Datum datum) {
            this->m_board[name].append(datum);
        }

        void clear(const name_type &name) {
            this->query(name).clear();
        }

        Datum max(const name_type &name) const {
            return this->query(name).max();
        }

        Datum min(const name_type &name) const {
            return this->query(name).min();
        }

        Datum avg(const name_type &name) const {
            return this->query(name).avg();
        }

        Datum sum(const name_type &name) const {
            return this->query(name).sum();
        }

        size_t count(const name_type &name) const {
            return this->query(name).count();
        }

        void clean() {
            this->m_board.clear();
        }

        auto begin() -> decltype(this->m_board.begin()) {
            return this->m_board.begin();
        }

        auto end() -> decltype(this->m_board.end()) {
            return this->m_board.end();
        }

        auto begin() const -> decltype(this->m_board.begin()) {
            return this->m_board.begin();
        }

        auto end() const -> decltype(this->m_board.end()) {
            return m_board.end();
        }

        value_type query(const name_type &name) const {
            auto it = this->m_board.find(name);
            if (it == this->m_board.end()) return Statistics<Datum>();
            return it->second;
        }

    private:
        std::unordered_map<name_type, value_type> m_board;
    };
}


#endif //TENSORSTACK_BOARD_BOARD_H
