#ifndef INTERVALBB_H_
#define INTERVALBB_H_

#include "evaluator.h"
#include "branching.h"
#include "pruning.h"
#include "ivm.h"

#include "libbounds.h"

class pbab;

template<typename T>
class Intervalbb{
public:
    Intervalbb(pbab* _pbb);

    bool initAtInterval(std::vector<int>& pos, std::vector<int>& end);
    void initFullInterval();
    void setRoot(const int* varOrder, int l1, int l2);

    void run();
    bool next();
    void clear();

    void setBest(const int);
    void eliminateJobs(std::vector<T> lb);
    bool boundLeaf(subproblem& node);

    static std::vector<T> rootRow;
    static int rootDir;
    static int first;

    virtual void boundAndKeepSurvivors(subproblem& subproblem,const int);
    virtual void boundAndKeepSurvivors_static(subproblem& subproblem,const int);

    void refineBounds(subproblem& node, const int be, std::vector<T>& lb,std::vector<T>& prio);

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
protected:
    pbab* pbb;
    int size;
    std::shared_ptr<ivm> IVM;

    long long int count_leaves;
    long long int count_decomposed;

    std::unique_ptr<Pruning> prune;
    std::unique_ptr<Branching> branch;
    std::unique_ptr<Evaluator<T>> eval;

    void unfold(int mode);
};

//static members
template<typename T>
std::vector<T> Intervalbb<T>::rootRow;
template<typename T>
int Intervalbb<T>::rootDir = 0;
template<typename T>
int Intervalbb<T>::first = true;


#endif
