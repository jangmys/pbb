#ifndef FSPNHOODS_H_
#define FSPNHOODS_H_

#include "libbounds.h"
#include "../neh/fastinsertremove.h"

#include <memory>

/// \brief neighborhoods based on the fast insert-remove operations
template<typename T>
class fspnhood{
public:
    explicit fspnhood(instance_abstract& inst) :
        m(std::make_unique<fastInsertRemove>(inst))
    {    };

    fspnhood(const std::vector<std::vector<int>> p_times,const int N, const int M) :
        m(std::make_unique<fastInsertRemove>(p_times,N,M)){ };

    std::unique_ptr<fastInsertRemove> m;

    int fastBREmove(std::vector<int>& perm, int pos);
    int fastBREmove(std::vector<int>& perm, int pos, int l1, int l2);

    int kImove(std::vector<int>& perm,int pos, int kmax);
    int fastkImove(std::vector<int>& perm,int kmax);
    int fastkImove(std::vector<int>& perm,int kmax,int l1,int l2);
};


#endif
