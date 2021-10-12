#include <atomic>
#include <iostream>

#include <pthread.h>

#ifndef STATISTICS_H_
#define STATISTICS_H_

struct statistics {
    statistics()
    {
        johnsonBounds = ATOMIC_VAR_INIT(0);
        simpleBounds  = ATOMIC_VAR_INIT(0);
        totDecomposed = ATOMIC_VAR_INIT(0);
        leaves        = ATOMIC_VAR_INIT(0);
    }

    std::atomic<uint64_t> johnsonBounds;
    std::atomic<uint64_t> simpleBounds;
    std::atomic<uint64_t> totDecomposed;
    std::atomic<uint64_t> leaves;

    void print()
    {
        std::cout<<"================="<<std::endl;
        std::cout<<"Exploration stats"<<std::endl;
        std::cout<<"================="<<std::endl;

        std::cout<<"TOT-BRANCHED:\t"<<totDecomposed<<std::endl;
        std::cout<<"TOT-LEAVES:\t"<<leaves<<std::endl;
    }
};

#endif
