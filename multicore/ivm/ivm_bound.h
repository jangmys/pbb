#ifndef IVM_BOUND_H
#define IVM_BOUND_H

#include <vector>

#include "branching.h"
#include "pruning.h"

class ivm;
class pbab;
class subproblem;

template<typename T>
class ivm_bound{
public:
    std::unique_ptr<pruning> prune;

    static std::vector<T> rootRow;
    static int rootDir;
    static int first;

    ivm_bound(pbab* _pbb);
    ~ivm_bound();

    void completeSchedule(const int job,const int order);

    // void allocate();
    void prepareSchedule(const ivm* IVM);

    void computeWeakBounds();
    void computeStrongBounds(const int be);


    void applyPruning(ivm* IVM, const int first,const int second);

    void weakBoundPrune(ivm* IVM);
    void strongBoundPrune(ivm* IVM);
    void mixedBoundPrune(ivm* IVM);

    void boundNode(const ivm* IVM);
    bool boundLeaf(ivm* IVM);
    void boundRootWeak(ivm *IVM);
    void boundRootStrong(ivm *IVM);
    void boundRoot(ivm* IVM);

    //get/set
    void getSchedule(int *sch);
private:
    enum boundtype{Weak,Strong};

    pbab* pbb;
    subproblem* node;

    std::vector<std::vector<T>> costs;

    std::vector<std::vector<T>> costsBegin;
    std::vector<std::vector<T>> costsEnd;

    std::vector<T> priorityBegin;
    std::vector<T> priorityEnd;

    int size;

    int eliminateJobs(ivm *IVM,std::vector<T> cost1,std::vector<T> cost2, std::vector<T> prio);

    void sortSiblingNodes(ivm* IVM);

    std::vector<std::unique_ptr<bound_abstract<T>>> bound;

    // bound_abstract<T> *bound[2];
    std::unique_ptr<branching> branch;
};


//static members
template<typename T>
std::vector<T> ivm_bound<T>::rootRow;
template<typename T>
int  ivm_bound<T>::rootDir = 0;
template<typename T>
int  ivm_bound<T>::first = true;



#endif /* IVM_BOUND_H */
