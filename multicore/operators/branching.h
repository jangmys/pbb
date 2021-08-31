#ifndef BRANCHING_H_
#define BRANCHING_H_

#include <limits.h>
#include <memory>

#define FRONT (0)
#define BACK (1)

class pbab;
class ivm;

class branching{
public:
    explicit branching(int _size) : size(_size)
    {};

    enum branchingDirection{Front,Back};

    virtual ~branching(){};
    virtual int operator()(const int*cb, const int* ce, const int line) = 0;

protected:
    int size;
};

//static : only left to right
class forwardBranching : public branching
{
public:
    explicit forwardBranching(int _size) : branching(_size){};

    //needs no argument...
    int operator()(const int*cb, const int* ce, const int line)
    {
        return Front;
    }
};

//static : only right to left
class backwardBranching : public branching
{
public:
    explicit backwardBranching(int _size) : branching(_size){};

    //needs no argument...
    int operator()(const int*cb, const int* ce, const int line)
    {
        return Back;
    }
};

//alternating : branching direction depends on depth
class alternateBranching final : public branching
{
public:
    explicit alternateBranching(int _size) : branching(_size){};

    int operator()(const int*cb, const int* ce, const int line)
    {
        return line%2?Back:Front;
    }
};

class maxSumBranching : public branching
{
public:
    explicit maxSumBranching(int _size) : branching(_size){};

    int operator()(const int*cb, const int* ce, const int line)
    {
        int sum = 0;

        for (int i = 0; i < size; ++i) {
            sum += (cb[i]-ce[i]);
        }

        return (sum < 0)?Back:Front; //choose larger sum, tiebreak : FRONT
    }
};

class minBranchBranching : public branching
{
public:
    explicit minBranchBranching(int _size,int UB = 0) : branching(_size),initialUB(UB)
    {};


    int operator()(const int*cb, const int* ce, const int line){
        int ub = initialUB;

        // int ub=local_best;
        // if(!arguments::singleNode)ub=initialUB;

        int elim = 0;
        int sum = 0;

        for (int i = 0; i < size; ++i) {
            if (cb[i]>=ub)elim++; //"pruned"
            else sum += cb[i];
            if (ce[i]>=ub)elim--; //"pruned"
            else sum -=ce[i];
        }
        //take set with lss open nodes / tiebreaker: greater average LB
        if(elim > 0)return Front;
        else if(elim < 0)return Back;
        else return (sum < 0)?Back:Front;
    }

private:
    int initialUB;
};


class minMinBranching : public branching
{
public:
    explicit minMinBranching(int _size,int UB) : branching(_size),initialUB(UB)
    {};

    int operator()(const int*cb, const int* ce, const int line){
        // int ub=local_best;
        // if(!arguments::singleNode)
        int ub=initialUB;

        int min=INT_MAX;
        int minCount=0;
        int elimCount=0;

        //find min lower bound
        for (int i = 0; i < size; ++i) {
            if(cb[i]<min && cb[i]>0)min=cb[i];
            if(ce[i]<min && ce[i]>0)min=ce[i];
        }
        //how many times lowest LB realized?
        for (int i = 0; i < size; ++i) {
            if(cb[i]==min)minCount++;
            if(ce[i]==min)minCount--;
            if (cb[i]>=ub)elimCount++;
            if (ce[i]>=ub)elimCount--;
        }
        //take set where min LB is realized LESS often
        if(minCount > 0)return Back;
        else if(minCount < 0)return Front;
        else return (elimCount > 0)?Front:Back;//break ties
    }
private:
    int initialUB;
};

static std::unique_ptr<branching> create_branching(int choice, int size, int initialUB)
{
    switch (choice) {
        case -3:{
            return std::make_unique<alternateBranching>(size);
        }
        case -2:{
            return std::make_unique<forwardBranching>(size);
        }
        case -1:{
            return std::make_unique<backwardBranching>(size);
        }
        case 1:{
            return std::make_unique<maxSumBranching>(size);
        }
        case 2:{
            return std::make_unique<minBranchBranching>(size,initialUB);
        }
        case 3:{
            return std::make_unique<minMinBranching>(size,initialUB);
        }
        default:{
            printf("branching rule not defined\n");
            break;
        }
    }
    return nullptr;
}


#endif
