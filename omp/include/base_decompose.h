#ifndef DECOMPOSE_H
#define DECOMPOSE_H

#include <vector>
#include <memory>

#include "libbounds.h"

template <class T>
class DecomposeBase
{
public:
    // DecomposeBase(){};
    DecomposeBase(std::unique_ptr<bound_abstract<int>> _eval):
        eval(std::move(_eval))
    {};

    virtual std::vector<std::unique_ptr<T>> operator()(T& n, const int best_ub) = 0;

    std::unique_ptr<bound_abstract<int>> eval;
};

#endif
