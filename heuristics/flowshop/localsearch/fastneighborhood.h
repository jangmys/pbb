#ifndef FSPNHOODS_H_
#define FSPNHOODS_H_

#include "libbounds.h"
#include "../neh/fastinsertremove.h"

#include <memory>

template<typename T>
class fspnhood{
public:
    fspnhood(instance_abstract* inst);
    ~fspnhood();

    std::unique_ptr<fastInsertRemove<T>> m;

    int N;

    int fastBREmove(int* perm, int pos);
    int fastBREmove(int* perm, int pos, int l1, int l2);

    int kImove(int* perm,int pos, int kmax);
    int fastkImove(int* perm,int kmax);
    int fastkImove(int* perm,int kmax,int l1,int l2);

};


#endif
