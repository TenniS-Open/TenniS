//
// Created by kier on 2019/3/4.
//

#include "run_test/walker.hpp"
#include "run_test/option.hpp"
#include "run_test/test_case.hpp"

#include <utils/box.h>

int main(int argc, const char *argv[]) {
    using namespace ts;

    if (argc < 2) {
        std::cerr << "Usage: <command> path [device [id]]" << std::endl;
        return 1;
    }
    std::string root = argv[1];

    std::string device = "cpu";
    int id = 0;

    if (argc > 2) {
        device = argv[2];
    }

    for (auto &ch : device) {
        ch = char(std::tolower(ch));
    }

    if (argc > 3) {
        id = int(std::strtol(argv[3], nullptr, 10));
    }

    ComputingDevice computing_device(device, id);

    auto subdirs = FindFlodersRecursively(root);

    int ok_count = 0;
    int failed_count = 0;

    for (auto &subdir : subdirs) {
        auto case_root = Join({root, subdir}, FileSeparator());
        auto case_filenames = FindFiles(case_root);
        TestCase tc;
        try {
            if (!tc.load(case_root, case_filenames)) {
                continue;
            }
        } catch (const Exception &e) {
            continue;
        }
        // run test
        std::cout << "==================== " << subdir << " on " << computing_device << " ====================" << std::endl;
        // try infer
        bool ok = false;
        try {
            ok = tc.run(computing_device, 100);
        } catch (const Exception &e) {
        }
        if (ok) {
            ok_count++;
            std::cout << "[OK]" << std::endl;
            // std::cout << tc.log();
        } else {
            failed_count++;
            std::cout << "[FAILED]" << std::endl;
            std::cout << tc.log();

        }
    }

    TS_LOG_INFO << "[OK]: " << ok_count << ", [FAILED]: " << failed_count;

    return 0;
}