#ifndef BRANCHING_H_
#define BRANCHING_H_

#include <limits.h>
#include <memory>

class pbab;

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

    explicit Branching(Branching::branchingType _type) : m_branchType(_type)
    {};

    virtual ~Branching(){};
    virtual int operator()(const int*cb, const int* ce, const int line){
        return Front;
    };

    virtual int pre_bound_choice(const int line){
        return Front;
    };

    virtual Branching::branchingType get_type()
    {
        return m_branchType;
    }

protected:
    Branching::branchingType m_branchType = Forward;
};

//static : only left to right
class forwardBranching : public Branching
{
public:
    explicit forwardBranching() :
        Branching(Branching::Forward){};

    //needs no argument...
    int operator()(const int*cb, const int* ce, const int line)
    {
        return Front;
    }

    int pre_bound_choice(const int line)
    {
        return Front;
    }
};

//static : only right to left
class backwardBranching : public Branching
{
public:
    explicit backwardBranching() : Branching(Branching::Backward){};

    //needs no argument...
    int operator()(const int*cb, const int* ce, const int line)
    {
        return Back;
    }

    int pre_bound_choice(const int line)
    {
        return Back;
    }
};

//alternating : Branching direction depends on depth
class alternateBranching final : public Branching
{
public:
    alternateBranching() : Branching(Branching::Bidirectional){};

    int operator()(const int*cb, const int* ce, const int line)
    {
        return (line%2)?Back:Front;
    }

    int pre_bound_choice(const int line)
    {
        return (line%2)?Back:Front;
    }
};

class maxSumBranching : public Branching
{
public:
    explicit maxSumBranching(int _size) : Branching(Branching::Bidirectional),size(_size){};

    int operator()(const int*cb, const int* ce, const int line)
    {
        int sum = 0;

        for (int i = 0; i < size; ++i) {
            sum += (cb[i]-ce[i]);
        }

        return (sum < 0)?Back:Front; //choose larger sum, tiebreak : FRONT
    }

    int pre_bound_choice(const int line)
    {
        return -1;
    }
private:
    int size;
};

class minBranchBranching : public Branching
{
public:
    minBranchBranching(int _size,int UB) : Branching(Branching::Bidirectional),size(_size),initialUB(UB)
    {};


    int operator()(const int*cb, const int* ce, const int line){
        int elim = 0;
        int sum = 0;

        for (int i = 0; i < size; ++i) {
            if (cb[i]>=initialUB)elim++; //"pruned"
            if (ce[i]>=initialUB)elim--; //"pruned"
            sum += (cb[i]-ce[i]);
        }
        //take set with lss open nodes / tiebreaker: greater average LB
        if(elim > 0)return Front;
        else if(elim < 0)return Back;
        else return (sum < 0)?Back:Front;
    }

    int pre_bound_choice(const int line)
    {
        return -1;
    }

private:
    int size;
    int initialUB;
};


class minMinBranching : public Branching
{
public:
    explicit minMinBranching(int _size, int _ub) : Branching(Branching::Bidirectional),size(_size),referenceUB(_ub)
    {};

    int operator()(const int*cb, const int* ce, const int line){
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
            if (cb[i]>=referenceUB)elimCount++;
            if (ce[i]>=referenceUB)elimCount--;
        }
        //take set where min LB is realized LESS often
        if(minCount > 0)return Back;
        else if(minCount < 0)return Front;
        else return (elimCount > 0)?Front:Back;//break ties
    }

    int pre_bound_choice(const int line)
    {
        return -1;
    }
private:
    int size;
    int referenceUB;
};

#endif
