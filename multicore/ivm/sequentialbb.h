#ifndef SEQUENTIALBB_H_
#define SEQUENTIALBB_H_

#include "ivm.h"
#include "ivm_bound.h"
// #include "../../common/include/ttime.h"

#include "libbounds.h"

class pbab;

class sequentialbb{
public:
    sequentialbb(pbab* _pbb, int _size);
    ~sequentialbb();

    bool initAtInterval(int * pos, int * end);
    void initFullInterval();
    void setRoot(const int* varOrder);
    bool solvedAtRoot();

    void run();
    void run(int* firstRow);

    bool next();
    void clear();

    void setBest(const int);

    ivm* IVM;

    int count_iters;
    int count_decomposed;

    ivm_bound<int>* bd;
protected:

    int size;

    void unfold(int mode);
};


#endif
