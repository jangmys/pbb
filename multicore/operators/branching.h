#ifndef BRANCHING_H_
#define BRANCHING_H_

#include <limits.h>
#include <memory>

class pbab;
class ivm;

class Branching{
public:
    // Used for array indexes!  Don't change the numbers!
    enum branchingDirection{
        Front = 0,
        Back = 1
    };
    enum branchingType{
        Forward = 0,
        Backward = 1,
        Bidirectional = 2
    };

    explicit Branching(int _size,Branching::branchingType _type) : size(_size),m_branchType(_type)
    {};

    virtual ~Branching(){};
    virtual int operator()(const int*cb, const int* ce, const int line) = 0;

    virtual Branching::branchingType get_type()
    {
        return m_branchType;
    }

protected:
    int size;
    Branching::branchingType m_branchType = Forward;
};

//static : only left to right
class forwardBranching : public Branching
{
public:
    explicit forwardBranching(int _size) :
        Branching(_size,Branching::Forward){};

    //needs no argument...
    int operator()(const int*cb, const int* ce, const int line)
    {
        return Front;
    }
};

//static : only right to left
class backwardBranching : public Branching
{
public:
    explicit backwardBranching(int _size) : Branching(_size,Branching::Backward){};

    //needs no argument...
    int operator()(const int*cb, const int* ce, const int line)
    {
        return Back;
    }
};

//alternating : Branching direction depends on depth
class alternateBranching final : public Branching
{
public:
    explicit alternateBranching(int _size) : Branching(_size,Branching::Bidirectional){};

    int operator()(const int*cb, const int* ce, const int line)
    {
        return (line%2)?Back:Front;
    }
};

class maxSumBranching : public Branching
{
public:
    explicit maxSumBranching(int _size) : Branching(_size,Branching::Bidirectional){};

    int operator()(const int*cb, const int* ce, const int line)
    {
        int sum = 0;

        for (int i = 0; i < size; ++i) {
            sum += (cb[i]-ce[i]);
        }

        return (sum < 0)?Back:Front; //choose larger sum, tiebreak : FRONT
    }
};

class minBranchBranching : public Branching
{
public:
    explicit minBranchBranching(int _size,int UB = 0) : Branching(_size,Branching::Bidirectional),initialUB(UB)
    {};


    int operator()(const int*cb, const int* ce, const int line){
        int ub = initialUB;

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


class minMinBranching : public Branching
{
public:
    explicit minMinBranching(int _size,int UB) : Branching(_size,Branching::Bidirectional),initialUB(UB)
    {};

    int operator()(const int*cb, const int* ce, const int line){
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

#endif
