//
// Created by kier on 2018/12/19.
//

#include "core/sink.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <map>

#include <immintrin.h>

int find_first_i(int i) {
    int p = 0;
    while (i) {
        ++p;
        i >>= 1;
    }
    return p;
};

void plot_list() {
    int i = 0;
    while (true) {
        auto p = find_first_i(i);
        std::cout << p << ", ";
        ++i;
        if (i % 16 == 0) std::cout << std::endl;
        if (p == 8) break;
    }
    std::cout << std::endl;
}

std::string to_bin_string(int32_t i) {
    std::ostringstream oss;
    oss << std::bitset<sizeof(int32_t)*8>(i);
    return oss.str();
}

void plot_block(const std::string &str, const std::vector<int> &blocks) {
    std::vector<std::pair<int, int>> ranges;
    int left = 0;
    for (auto &block : blocks) {
        ranges.emplace_back(std::make_pair(left, block));
        left += block;
    }
    bool first = true;
    for (auto &range : ranges) {
        if (first) first = false;
        else std::cout << " ";
        std::cout << str.substr(range.first, range.second);
    }
    std::cout << std::endl;
}


void test() {
    int count = 0;
    int match = 0;

    int i = 0;
    while (true) {
        if (i & 0xffffff00) break;

        int Q = 0;

        auto s = uint8_t(i & 0xff);
        while (Q < 8) {
            auto f = ts::number_converter<float, uint8_t>::ToCommon(s, Q);
            auto ts = ts::number_converter<float, uint8_t>::ToFixed(f, Q);

            ++count;
            if (ts == s) {
                ++match;
            } else {
                std::cerr << "Missmatch sink(Q=" << Q <<"): 0x" << std::hex << uint32_t(s) << " vs. 0x" << uint32_t(ts)<< std::oct << std::endl;
            }

            ++Q;
        }

        ++i;
    }

    std::cout << "count=" << count << ", match=" << match << std::endl;
}

int8_t random_sink8() {
    return int8_t (rand() & 0xff);
}

int main() {
    // plot_list();

    std::cout << "Converting each sink8 to float then convert back." << std::endl;

    test();

    ts::sink8<4> s = 3.5;

    float f = s;

    std::cout << f << std::endl;


    return 0;
}