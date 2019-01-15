//
// Created by kier on 2018/12/21.
//

#include <omp.h>
#include <cstdio>

#include "runtime/runtime.h"
#include "runtime/inside/parallel.h"
#include "kernels/common/simd.h"
#include "kernels/common/openmp.h"
#include "utils/platform.h"
#include <utils/random.h>

#if TS_PLATFORM_OS_MAC || TS_PLATFORM_OS_IOS
#include <Accelerate/Accelerate.h>

#elif TS_PLATFORM_OS_LINUX
#include <openblas/cblas.h>
#elif TS_PLATFORM_OS_WINDOWS && TS_PLATFORM_CC_MINGW
#include <OpenBLAS/cblas.h>
#else
#incldue <cblas.h>
#endif

template <typename T>
class TestCase {
public:
    int N = 0;
    std::shared_ptr<T> x;
    std::shared_ptr<T> y;

    T result = 0;

    void update();

    static TestCase Random(ts::Random &rand, int N = 0) {
        if (N  == 0) N = rand.next(1000, 10000);

        TestCase test;
        test.N = N;
        test.x.reset(new T[N], std::default_delete<T[]>());
        test.y.reset(new T[N], std::default_delete<T[]>());

        for (int i = 0; i < N; ++i) test.x.get()[i] = rand.next(-100, 100) / 100.0;
        for (int i = 0; i < N; ++i) test.y.get()[i] = rand.next(-100, 100) / 100.0;

        test.update();

        return test;
    }
};

template <>
void TestCase<float>::update() {
    this->result = cblas_sdot(N, x.get(), 1, y.get(), 1);
}

template <>
void TestCase<double>::update() {
    this->result = cblas_ddot(N, x.get(), 1, y.get(), 1);
}

using dot_function = std::function<float(const float *, const float *, int)>;

void test_diff(TestCase<float> &test, const dot_function &dot, const std::string &header = "") {
    std::ostringstream oss;
    auto diff = std::fabs(test.result - dot(test.x.get(), test.y.get(), test.N));
    oss << header << "diff = " << diff << std::endl;
    std::cout << oss.str();
}

void test_threads(TestCase<float> &test, const dot_function &dot, int threads, int times) {
    std::vector<std::shared_ptr<std::thread>> pool;

    for (int i = 0; i < threads; ++i) {
        std::string header = std::string("Thread-") + std::to_string(i) + ": ";
        pool.emplace_back(std::make_shared<std::thread>([=, &test]() {
            for (int i = 0; i < times; ++i) {
                test_diff(test, dot, header);
            }
        }));
    }
    for (auto &t : pool) {
        t->join();
    }
}

float dot0(const float *x, const float *y, int N) {
    return cblas_sdot(N, x, 1, y, 1);
}

float dot1(const float *x, const float *y, int N) {
    float sum = 0;
    int i = 0;

    for (i = 0; i < N; ++i) {
        sum += *x++ * *y++;
    }

    return sum;
}

inline float dot2(const float *x, const float *y, int N) {
    float sum = 0;
    int i = 0;

    for (i = 0; i < N - 3; i += 4) {
        sum += *x++ * *y++;
        sum += *x++ * *y++;
        sum += *x++ * *y++;
        sum += *x++ * *y++;
    }

    for (; i < N; ++i) {
        sum += *x++ * *y++;
    }

    return sum;
}

float dot3(const float *x, const float *y, int N) {
    std::vector<float> parallel_sum(TS_PARALLEL_SIZE, 0);
    TS_PARALLEL_RANGE_BEGIN(range, 0, N)
            const float *xx = x + range.first;
            const float *yy = y + range.first;
            const auto count = range.second - range.first;
            parallel_sum[__parallel_id] += dot2(xx, yy, count);
    TS_PARALLEL_RANGE_END()
    float sum = 0;
    for (auto value : parallel_sum) sum += value;
    return sum;
}

float dot3x(const float *x, const float *y, int N) {
    std::vector<float> parallel_sum(TS_PARALLEL_SIZE, 0);
    TS_PARALLEL_FOR_BEGIN(i, 0, N)
                parallel_sum[__parallel_id] += x[i] * y[i];
    TS_PARALLEL_FOR_END()
    float sum = 0;
    for (auto value : parallel_sum) sum += value;
    return sum;
}

float dot4(const float *x, const float *y, int N) {
    float sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(4)
    for (int i = 0; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}


float dot5(const float *x, const float *y, int N) {
    float sum = 0;
    int i = 0;

    ts::float32x4 sumx4 = 0;

    for (i = 0; i < N - 3; i += 4) {
        sumx4 += ts::float32x4(x) * ts::float32x4(y);
        x += 4;
        y += 4;
    }

    sum = ts::sum(sumx4);

    for (; i < N; ++i) {
        sum += *x++ * *y++;
    }

    return sum;
}

float dot6(const float *x, const float *y, int N) {
    std::vector<float> parallel_sum(TS_PARALLEL_SIZE, 0);
    TS_PARALLEL_RANGE_BEGIN(range, 0, N)
            const float *xx = x + range.first;
            const float *yy = y + range.first;
            const auto count = range.second - range.first;
            parallel_sum[__parallel_id] += dot5(xx, yy, count);
    TS_PARALLEL_RANGE_END()
    float sum = 0;
    for (auto value : parallel_sum) sum += value;
    return sum;
}

float dot7(const float *x, const float *y, int N) {
    float sum = 0;
    ts::float32x4 sumx4 = 0;
#pragma omp parallel for reduction(+:sumx4) num_threads(ts::openmp_threads(N / 4))
    for (int i = 0; i < N - 3; i += 4) {
        sumx4 += ts::float32x4(&x[i]) * ts::float32x4(&y[i]);
    }

    sum = ts::sum(sumx4);

    for (int i = N / 4 * 4; i < N; ++i) {
        sum += x[i] * y[i];
    }

    return sum;
}

float dot8(const float *x, const float *y, int N) {
    float sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(ts::openmp_threads(N / 4))
    for (int i = 0; i < N - 3; i += 4) {
        ts::float32x4 sumx4 = ts::float32x4(&x[i]) * ts::float32x4(&y[i]);
        sum += ts::sum(sumx4);
    }

    for (int i = N / 4 * 4; i < N; ++i) {
        sum += x[i] * y[i];
    }

    return sum;
}

void test_loop_bottom(int top, int top_id) {
#pragma omp parallel for num_threads(1)
    for (int i = 0; i < 10; ++i) {
        printf("Top task: %2d, Top ID: %d, Bottom task: %2d, Bottom ID: %d\n", top, top_id, i, omp_get_thread_num());
    }
}

void test_loop_top() {
#pragma omp parallel for num_threads(1)
    for (int i = 0; i < 10; ++i) {
        test_loop_bottom(i, omp_get_thread_num());
    }
}

void print_avg_time(const std::string &title, const int times, dot_function func, const float *a, const float *b, int N) {
    using namespace std::chrono;
    microseconds duration(0);

    float sum = 0;

    auto start = system_clock::now();

    for (int i = 0; i < times; ++i) {
        sum += func(a, b, N);
    }

    sum /= times;

    auto end = system_clock::now();
    duration += duration_cast<microseconds>(end - start);
    double spent = 1.0 * duration.count() / 1000;

    std::cout << title << ": sum=" << sum << ", spent=" << spent << "ms" << std::endl;
}

int main()
{
    ts::RuntimeContext runtime;
    runtime.set_computing_thread_number(4);

    ts::ctx::bind<ts::ThreadPool> _bind_thread_pool(runtime.thread_pool());
    ts::ctx::bind<ts::RuntimeContext> _bind_runtime(runtime);

    srand(4482);

    static const int times = 1000;
    static const int N = 102400;
    float a[N], b[N];
    for (int i = 0; i < N; ++i) {
        a[i] = rand() % 400 / 100.0 - 2;
        b[i] = rand() % 400 / 100.0 - 2;
    }

    print_avg_time("BLAS        ", times, dot0, a, b, N);
    print_avg_time("Pure CPU    ", times, dot1, a, b, N);
    print_avg_time("Pure CPU(4) ", times, dot2, a, b, N);
    print_avg_time("Threads CPU ", times, dot3, a, b, N);
    // print_avg_time("Threads CPU(x) ", times, dot3x, a, b, N);
    print_avg_time("OpenMP CPU  ", times, dot4, a, b, N);
    print_avg_time("Pure SIMD   ", times, dot5, a, b, N);
    print_avg_time("Threads SIMD", times, dot6, a, b, N);
    print_avg_time("OpenMP SIMD ", times, dot7, a, b, N);
    print_avg_time("OpenMP SIMD2", times, dot8, a, b, N);

    std::cout << "========== test stable in multi threads ==============" << std::endl;

    ts::Random rand(4831);
    auto test = TestCase<float>::Random(rand);
    test_threads(test, dot7, 4, 10);

    return 0;
}