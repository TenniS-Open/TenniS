//
// Created by kier on 2018/10/31.
//

#include "utils/box.h"
#include "utils/platform.h"

#include <algorithm>
#include <memory>
#include <cmath>
#include <cfenv>
#include <ctime>
#include <iomanip>
#include <sstream>


namespace ts {
    template<typename T>
    static inline T min(T a, T b, T c) {
        return std::min<T>(std::min<T>(a, b), c);
    }

    int edit_distance(const std::string &lhs, const std::string &rhs) {
        const size_t M = lhs.length();  // rows
        const size_t N = rhs.length();  // cols

        if (M == 0) return int(N);
        if (N == 0) return int(M);

        std::unique_ptr<int[]> dist(new int[M * N]);
#define __EDIT_DIST(m, n) (dist[(m) * N + (n)])
        __EDIT_DIST(0, 0) = lhs[0] == rhs[0] ? 0 : 2;
        for (size_t n = 1; n < N; ++n) {
            __EDIT_DIST(0, n) = __EDIT_DIST(0, n - 1) + 1;
        }
        for (size_t m = 1; m < M; ++m) {
            __EDIT_DIST(m, 0) = __EDIT_DIST(m - 1, 0) + 1;
        }
        for (size_t m = 1; m < M; ++m) {
            for (size_t n = 1; n < N; ++n) {
                if (lhs[m] == rhs[n]) {
                    __EDIT_DIST(m, n) = min(
                            __EDIT_DIST(m - 1, n),
                            __EDIT_DIST(m, n - 1),
                            __EDIT_DIST(m - 1, n - 1));
                } else {
                    __EDIT_DIST(m, n) = min(
                            __EDIT_DIST(m - 1, n) + 1,
                            __EDIT_DIST(m, n - 1) + 1,
                            __EDIT_DIST(m - 1, n - 1) + 2);
                }
            }
        }
        return dist[M * N - 1];
#undef __EDIT_DIST
    }

    static int safe_ceil(double x) {
        int result;
        int save_round = std::fegetround();
        std::fesetround(FE_UPWARD);
        result = int(std::lrint(x));
        std::fesetround(save_round);
        return result;
    }

    std::vector<std::pair<int, int>> split_bins(int first, int second, int bins) {
        if (second <= first) return {};
        if (bins <= 1) return {{first, second}};
        auto step = safe_ceil((double(second) - double(first)) / bins);
        if (step < 1) step = 1;
        auto anchor = first;

        std::vector<std::pair<int, int>> result_bins;
        while (anchor + step < second) {
            result_bins.emplace_back(std::make_pair(anchor, anchor + step));
            anchor += step;
        }
        if (anchor < second) {
            result_bins.emplace_back(std::make_pair(anchor, second));
        }
        return result_bins;
    }

    static long safe_lceil(double x) {
        long result;
        int save_round = std::fegetround();
        std::fesetround(FE_UPWARD);
        result = long(std::lrint(x));
        std::fesetround(save_round);
        return result;
    }

    std::vector<std::pair<size_t, size_t>> lsplit_bins(size_t first, size_t second, size_t bins) {
        if (second <= first) return {};
        if (bins <= 1) return {{first, second}};
        auto step = safe_lceil((double(second) - double(first)) / bins);
        if (step < 1) step = 1;
        auto anchor = first;

        std::vector<std::pair<size_t, size_t >> result_bins;
        while (anchor + step < second) {
            result_bins.emplace_back(std::make_pair(anchor, anchor + step));
            anchor += step;
        }
        if (anchor < second) {
            result_bins.emplace_back(std::make_pair(anchor, second));
        }
        return result_bins;
    }

    static struct tm time2tm(std::time_t from) {
        std::tm to = {0};
#if TS_PLATFORM_CC_MSVC || TS_PLATFORM_CC_MINGW
        localtime_s(&to, &from);
#else
        localtime_r(&from, &to);
#endif
        return to;
    }

    std::string to_string(time_point tp, const std::string &format) {
        std::time_t tt = std::chrono::system_clock::to_time_t(tp);
        std::tm even = time2tm(tt);
        char tmp[64];
        std::strftime(tmp, sizeof(tmp), format.c_str(), &even);
        return std::string(tmp);
    }

    std::string now_time(const std::string &format) {
        return to_string(std::chrono::system_clock::now(), format);
    }

    std::string::size_type FindDecollator(const std::string &str, const std::string &sep, std::string::size_type off) {
        if (off == std::string::npos) return std::string::npos;
        std::string::size_type i = off;
        for (; i < str.length(); ++i) {
            if (sep.find(str[i]) != std::string::npos) return i;
        }
        return std::string::npos;
    }

    std::vector<std::string> Split(const std::string &str, const std::string &sep, size_t _size) {
        std::vector<std::string> result;
        std::string::size_type left = 0, right;

        result.reserve(_size);
        while (true) {
            right = FindDecollator(str, sep, left);
            result.push_back(str.substr(left, right == std::string::npos ? std::string::npos : right - left));
            if (right == std::string::npos) break;
            left = right + 1;
        }
        return std::move(result);
    }

    std::string Join(const std::vector<std::string>& list, const std::string &sep) {
        std::ostringstream oss;
        for (size_t i = 0; i < list.size(); ++i) {
            if (i) oss << sep;
            oss << list[i];
        }
        return oss.str();
    }

    std::string memory_size_string(uint64_t memory_size) {
        static const char *base[] = {"B", "KB", "MB", "GB", "TB"};
        static const uint64_t base_size = sizeof(base) / sizeof(base[0]);
        auto number = double(memory_size);
        uint64_t base_time = 0;
        while (number >= 1024.0 && base_time + 1 < base_size) {
            number /= 1024.0;
            base_time++;
        }
        number = std::round(number * 10.0) / 10.0;
        std::ostringstream oss;
        if (uint64_t(number * 10) % 10 == 0) {
            oss << std::fixed << std::setprecision(0) << number << base[base_time];
        } else {
            oss << std::fixed << std::setprecision(1) << number << base[base_time];
        }
        return oss.str();
    }
}
