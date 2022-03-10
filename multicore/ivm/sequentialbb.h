#ifndef SEQUENTIALBB_H_
#define SEQUENTIALBB_H_

#include "evaluator.h"
#include "branching.h"
#include "pruning.h"
#include "ivm.h"

#include "libbounds.h"

class pbab;

template<typename T>
class sequentialbb{
public:
    sequentialbb(pbab* _pbb, int _size);

    bool initAtInterval(std::vector<int>& pos, std::vector<int>& end);
    void initFullInterval();
    void setRoot(const int* varOrder);

    void run();
    bool next();
    void clear();

    void setBest(const int);
    void eliminateJobs(std::vector<T> lb);
    bool boundLeaf();

    static std::vector<T> rootRow;
    static int rootDir;
    static int first;

    void weakBoundPrune();
    void mixedBoundPrune();
    void strongBoundPrune();

    void boundNode(std::shared_ptr<ivm> IVM, std::vector<T>& lb_begin, std::vector<T>& lb_end);
    void computeStrongBounds(subproblem& node, const int be, std::vector<T>& lb);

    long long int get_leaves_count() const
    {
        return count_leaves;
    }
    long long int get_decomposed_count() const
    {
        return count_decomposed;
    }
    void reset_node_counter(){
        count_leaves = 0;
        count_decomposed = 0;
    }


    std::shared_ptr<ivm> get_ivm(){
        return IVM;
    }
private:
    pbab* pbb;
    int size;
    std::shared_ptr<ivm> IVM;

    long long int count_leaves{0};
    long long int count_decomposed;

    std::vector<T> lower_bound_begin;
    std::vector<T> lower_bound_end;
    std::vector<T> priority_begin;
    std::vector<T> priority_end;

    std::unique_ptr<Pruning> prune;
    std::unique_ptr<branching> branch;
    std::unique_ptr<evaluator<T>> eval;

    void unfold(int mode);
};

//static members
template<typename T>
std::vector<T> sequentialbb<T>::rootRow;
template<typename T>
int sequentialbb<T>::rootDir = 0;
template<typename T>
int sequentialbb<T>::first = true;


#endif
