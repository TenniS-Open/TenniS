//
// Created by Lby on 2017/8/23.
//

#include "utils/except.h"
#include "vat.h"
#include <algorithm>

namespace ts {

    Vat::Vat() {
    }

    Vat::Vat(const Pot::allocator &ator)
        : m_allocator(ator) {
    }

    void *Vat::malloc(size_t _size) {
        // find first small piece
        Pot pot(m_allocator);
        if (!m_heap.empty())
        {
            size_t i = 0;
            for (; i < m_heap.size() - 1; ++i)
            {
                if (m_heap[i].capacity() >= _size) break;
            }
            pot = m_heap[i];
            m_heap.erase(m_heap.begin() + i);
        }
        void *ptr = pot.malloc(_size);
        m_dict.insert(std::pair<void *, Pot>(ptr, pot));

        return ptr;
    }

    void Vat::free(const void *ptr) {
        auto key = const_cast<void *>(ptr);
        auto it = m_dict.find(key);
        if (it == m_dict.end()) {
            throw Exception("Can not free this ptr");
        }

        auto &pot = it->second;

        auto ind = m_heap.begin();
        while (ind != m_heap.end() && ind->capacity() < pot.capacity()) ++ind;
        m_heap.insert(ind, pot);

        m_dict.erase(key);
    }

    void Vat::reset() {
        for (auto &pair : m_dict) {
            m_heap.push_back(pair.second);
        }
        m_dict.clear();
        std::sort(m_heap.begin(), m_heap.end(), [](const Pot &p1, const Pot &p2){return p1.capacity() < p2.capacity(); });
    }

    void Vat::dispose() {
        m_dict.clear();
        m_heap.clear();
    }

    void Vat::swap(Vat &that)
    {
        this->m_heap.swap(that.m_heap);
        this->m_dict.swap(that.m_dict);
    }

    Vat::Vat(Vat &&that)
    {
        this->swap(that);
    }

    Vat &Vat::operator=(Vat &&that)
    {
        this->swap(that);
        return *this;
    }
}
