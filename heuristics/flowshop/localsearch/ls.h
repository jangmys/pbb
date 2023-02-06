#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include "fastneighborhood.h"
#include "libbounds.h"

class LocalSearch{
public:
    LocalSearch(instance_abstract& inst);

    int operator()(std::vector<int>& perm, int l1, int l2);

    int localSearchBRE(std::vector<int>& perm, int l1, int l2);
    int localSearchKI(std::vector<int>& perm,const int kmax);

    std::unique_ptr<fspnhood<int>> nhood;
};


#endif
