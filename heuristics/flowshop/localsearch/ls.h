#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "fastneighborhood.h"
#include "libbounds.h"

#include <vector>
#include <array>

class LocalSearchBase{
public:
    LocalSearchBase(){};
};

class LocalSearch : public LocalSearchBase{
public:
    LocalSearch(instance_abstract& _inst) :
        LocalSearchBase(),nhood(std::make_unique<fspnhood<int>>(_inst))
    {};

    LocalSearch(const std::vector<std::vector<int>> p_times, const int N, const int M) :
        nhood(std::make_unique<fspnhood<int>>(p_times,N,M))
    {};

    int operator()(std::vector<int>& perm, int l1, int l2);

    int localSearchBRE(std::vector<int>& perm);
    int localSearchBRE(std::vector<int>& perm, int l1, int l2);

    int localSearchKI(std::vector<int>& perm,const int kmax);

    std::unique_ptr<fspnhood<int>> nhood;
};

#endif
