#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "fastneighborhood.h"
#include "libbounds.h"


class LocalSearchBase{
public:
    LocalSearchBase(){};
};


class LocalSearch : LocalSearchBase{
public:
    LocalSearch(instance_abstract& inst);
    LocalSearch(const std::vector<std::vector<int>> p_times, const int N, const int M);

    int operator()(std::vector<int>& perm, int l1, int l2);

    int localSearchBRE(std::vector<int>& perm);
    int localSearchBRE(std::vector<int>& perm, int l1, int l2);

    int localSearchKI(std::vector<int>& perm,const int kmax);

    std::unique_ptr<fspnhood<int>> nhood;
};


#endif
