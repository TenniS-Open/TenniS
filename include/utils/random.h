//
// Created by kier on 2018/7/19.
//

#ifndef TENSORSTACK_UTILS_RANDOM_H
#define TENSORSTACK_UTILS_RANDOM_H

namespace ts {
    class MT19937 {
    public:
        MT19937();

        explicit MT19937(int __seed);

        void srand(int __seed);

        int rand();

        static const int MAX = 0x7fffffff;     // 2 ^ 31 - 1
    private:
        static const int N = 624;      //624 * 32 - 31 = 19937
        int MT[N];
        int m_i = 0;
        int m_seed;
    };

    class Random {
    public:

        Random();

        explicit Random(int __seed);

        // 设置随机数种子
        void seed(int __seed);

        // 获取[0, MT19937::MAX]的平均分布随机数
        int next();

        // 获取[min, max]的平均分布随机数
        int next(int min, int max);

        // 获得[0, 1]的平均分布随机数
        double u();

        // 根据p概率返回真值
        bool binomial(double p);

        // 指数分布随机数
        double exp(double beta);

        // 瑞利分布随机数
        double ray(double mu);

        // 韦布尔分布随机数
        double weibull(double alpha, double beta);

        // N(0, 1)正态分布随机数
        double normal();

        // N(mu, delta^2)正态分布随机数
        double normal(double mu, double delta);

    private:
        MT19937 mt;
    };

    extern Random random;
}


#endif //TENSORSTACK_UTILS_RANDOM_H
