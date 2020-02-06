//
// Created by kier on 2020/2/3.
//

#ifndef TENNIS_UTILS_OTL_H
#define TENNIS_UTILS_OTL_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#include <string>
#include <sstream>

#include "except.h"
#include "log.h"

namespace ts {
    namespace otl {
        static inline unsigned int elf_hash(const char *str) {
            unsigned int hash = 0;
            unsigned int x = 0;
            while (*str) {
                hash = (hash << 4) + *str;
                if ((x = hash & 0xf0000000) != 0) {
                    hash ^= (x >> 24);
                    hash &= ~x;
                }
                str++;
            }
            return (hash & 0x7fffffff);
        }

        namespace sso {
            class OutOfRangeException : Exception {
            public:
                OutOfRangeException(size_t N, const std::string &std_string)
                        : Exception(Message(N, std_string)) {}

                static std::string Message(size_t N, const std::string &std_string) {
                    std::ostringstream oss;
                    oss << "Can not convert \"" << std_string << "\" (" << std_string.length() << ") to "
                        << "otl::sso::string<" << N << ">";
                    return oss.str();
                }
            };

            template<size_t N>
            class string {
            private:
                char m_buf[N] = {'\0'};
            public:
                using self = string;
                constexpr static size_t CAPACITY = N - 1;

                using iterator = char *;
                using const_iterator = const char *;

                string() {
                    // std::memset(m_buf, 0, N);
                }

                string(const string &other) = default;

                string(const std::string &std_string)
                        : self() {
                    if (std_string.length() > CAPACITY) {
                        TS_LOG_ERROR << OutOfRangeException::Message(N, std_string) << eject;
                    }
                    std::strcpy(m_buf, std_string.c_str());
                }

                string(const char *c_str)
                        : self() {
                    if (c_str == nullptr) return;
                    if (std::strlen(c_str) > CAPACITY) {
                        TS_LOG_ERROR << OutOfRangeException::Message(N, c_str) << eject;
                    }
                    std::strcpy(m_buf, c_str);
                }

                string(std::nullptr_t)
                        : self() {}

                const char *c_str() const { return m_buf; }

                size_t length() const {
                    return std::strlen(m_buf);
                }

                const char *data() const { return m_buf; }

                size_t size() const { return CAPACITY; }

                bool empty() const { return m_buf[0] == '\0'; }

                iterator begin() { return m_buf; }

                iterator end() { return m_buf + this->length(); }

                const_iterator begin() const { return m_buf; }

                const_iterator end() const { return m_buf + this->length(); }

                std::string std() const { return std::string(c_str()); }
            };

            template<size_t N1, size_t N2>
            inline bool operator==(const string<N1> &lhs, const string<N2> &rhs) {
                return std::strcmp(lhs.c_str(), rhs.c_str()) == 0;
            }

            template<size_t N1, size_t N2>
            inline bool operator!=(const string<N1> &lhs, const string<N2> &rhs) {
                return !operator==(lhs, rhs);
            }

            template<size_t N1, size_t N2>
            inline bool operator>(const string<N1> &lhs, const string<N2> &rhs) {
                return std::strcmp(lhs.c_str(), rhs.c_str()) > 0;
            }

            template<size_t N1, size_t N2>
            inline bool operator<(const string<N1> &lhs, const string<N2> &rhs) {
                return std::strcmp(lhs.c_str(), rhs.c_str()) < 0;
            }

            template<size_t N1, size_t N2>
            inline bool operator>=(const string<N1> &lhs, const string<N2> &rhs) {
                return !operator<(lhs, rhs);
            }

            template<size_t N1, size_t N2>
            inline bool operator<=(const string<N1> &lhs, const string<N2> &rhs) {
                return !operator>(lhs, rhs);
            }

            template<size_t N>
            inline std::ostream &operator<<(std::ostream &out, const string<N> &str) {
                return out << str.c_str();
            }

            template<>
            inline string<8>::string() {
                std::memset(m_buf, 0, 8);
            }

            template<>
            inline bool operator==(const string<8> &lhs, const string<8> &rhs) {
                return *reinterpret_cast<const uint64_t *>(lhs.data())
                       == *reinterpret_cast<const uint64_t *>(rhs.data());
            }

            template<>
            inline bool operator>(const string<8> &lhs, const string<8> &rhs) {
                return *reinterpret_cast<const uint64_t *>(lhs.data())
                       > *reinterpret_cast<const uint64_t *>(rhs.data());
            }

            template<>
            inline bool operator<(const string<8> &lhs, const string<8> &rhs) {
                return *reinterpret_cast<const uint64_t *>(lhs.data())
                       < *reinterpret_cast<const uint64_t *>(rhs.data());
            }
        }

        template <size_t N>
        using string = sso::string<N>;

        class VectorOutOfRangeException : Exception {
        public:
            VectorOutOfRangeException(size_t N, int i)
                    : Exception(Message(N, i)) {}

            static std::string Message(size_t N, int i) {
                std::ostringstream oss;
                oss << "Index " << i << " out of range of "<< "otl::vector<" << N << ">";
                return oss.str();
            }
        };

        template <typename T, size_t N, typename S = size_t>
        class vector {
        private:
            T m_buf[N];
            S m_size = 0;
        public:
            using self = vector;
            using value_type = T;
            using size_type = size_t;

            constexpr static size_t CAPACITY = N;

            using iterator = T *;
            using const_iterator = const T *;

            // copy
            explicit vector() = default;

            // from STL
            vector(const std::vector<T> &std_vector)
                : self(std_vector.begin(), std_vector.end()) {}

            // fill
            explicit vector (size_type n)
                : m_size(S(n)) {}

            vector (size_type n, const value_type& val)
                : m_size(S(n)) {
                for (auto &slot : *this) slot = val;
            }

            // range
            template <class InputIterator>
            vector(InputIterator first, InputIterator last) {
                size_type n = 0;
                for (auto it = first; it != last; ++it) {
                    if (n > CAPACITY) {
                        TS_LOG_ERROR << VectorOutOfRangeException::Message(N, n) << eject;
                    }
                    m_buf[n] = *it;
                    ++n;
                }
                m_size = S(n);
            }

            // copy
            vector(const vector& x) = default;

            // initializer list
            vector(std::initializer_list<value_type> il) {
                if (il.size() > CAPACITY) {
                    TS_LOG_ERROR << VectorOutOfRangeException::Message(N, il.size()) << eject;
                }
                size_type n = 0;
                for (auto it = il.begin(); it != il.end(); ++it) {
                    m_buf[n] = *it;
                    ++n;
                }
                m_size = S(n);
            }

            iterator begin() { return m_buf; }

            iterator end() { return m_buf + this->length(); }

            const_iterator begin() const { return m_buf; }

            const_iterator end() const { return m_buf + this->length(); }

            T *data() { return m_buf; }

            const T *data() const { return m_buf; }

            size_type size() const { return size_type(m_size); }

            size_type capacity() const { return CAPACITY; }
        };

        template <typename T, size_t N, typename S>
        inline std::ostream &operator<<(std::ostream &out, const vector<T, N, S> &vec) {
            using size_type = typename vector<T, N, S>::size_type;
            std::ostringstream oss;
            oss << "[";
            for (size_type i = 0; i < vec.size(); ++i) {
                if (i) oss << ", ";
                oss << i;
            }
            oss << "]";
            return out << oss.str();
        }
    }
}

namespace std {
    template<size_t N>
    struct hash<ts::otl::sso::string<N>> {
        using self = ts::otl::sso::string<N>;
        std::size_t operator()(const self &key) const {
            using std::size_t;
            using std::hash;

            return size_t(ts::otl::elf_hash(key.c_str()));
        }
    };
}

#endif //TENNIS_UTILS_OTL_H
