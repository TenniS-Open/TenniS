//
// Created by kier on 2019/1/29.
//

#include <board/board.h>

#include <iostream>

int main() {
    using namespace ts;
    Board<float> board;
    board.append("A", 1);
    board.append("B", 3);
    board.append("A", 3);
    board.append("B", 9);

    for (auto &name_value : board) {
        std::cout << name_value.first << ": ";
        for (auto &datum : name_value.second) {
            std::cout << datum << ", ";
        }
        std::cout << "avg = " << name_value.second.avg() << ".";
        std::cout << std::endl;
    }

    return 0;
}

