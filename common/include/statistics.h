#include <atomic>
#include <pthread.h>

#ifndef STATISTICS_H_
#define STATISTICS_H_

struct statistics {
    std::atomic<uint64_t> johnsonBounds;
    std::atomic<uint64_t> simpleBounds;
    std::atomic<uint64_t> totDecomposed;
    std::atomic<uint64_t> leaves;
};

#endif
