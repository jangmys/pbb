#ifndef OPERATOR_FACTORY_H
#define OPERATOR_FACTORY_H

#include "arguments.h"

// #include "evaluator.h"
#include "branching.h"
#include "pruning.h"

#include "libbounds.h"


class PruningFactoryInterface
{
public:
    virtual std::unique_ptr<Pruning> make_pruning() = 0;
};

class PruneLargerEqualFactory : public PruningFactoryInterface
{
public:
    std::unique_ptr<Pruning> make_pruning() override
    {
        return std::make_unique<keepSmaller>();
    }
};

class PruneLargerFactory : public PruningFactoryInterface
{
public:
    std::unique_ptr<Pruning> make_pruning() override
    {
        return std::make_unique<keepEqualOrSmaller>();
    }
};

//===================================================================
//===================================================================
//===================================================================

class BranchingFactoryInterface
{
public:
    BranchingFactoryInterface(int choice, int size, int initialUB) : _choice(choice),_size(size),_initialUB(initialUB){};

    virtual std::unique_ptr<Branching> make_branching() = 0;

protected:
    int _choice;
    int _size;
    int _initialUB;
};

class PFSPBranchingFactory : public BranchingFactoryInterface
{
public:
    PFSPBranchingFactory(int choice, int size, int initialUB) : BranchingFactoryInterface(
        choice,size,initialUB
    ){};


    std::unique_ptr<Branching> make_branching() override
    {
        switch (_choice) {
            case -3:{
                return std::make_unique<alternateBranching>(_size);
            }
            case -2:{
                return std::make_unique<forwardBranching>(_size);
            }
            case -1:{
                return std::make_unique<backwardBranching>(_size);
            }
            case 1:{
                return std::make_unique<maxSumBranching>(_size);
            }
            case 2:{
                return std::make_unique<minBranchBranching>(_size,_initialUB);
            }
            case 3:{
                return std::make_unique<minMinBranching>(_size,_initialUB);
            }
            default:{
                printf("branching rule not defined\n");
                break;
            }
        }
        return nullptr;
    }
};


//===================================================================
//===================================================================
//===================================================================

// template<typename T>
// class BoundFactoryInterface
// {
// public:
//     virtual std::unique_ptr<bound_abstract<T>> make_bound(std::unique_ptr<instance_abstract>& inst, int bound_mode) = 0;
// };


class OperatorFactory
{
  public:
    static std::unique_ptr<Branching> createBranching(int choice, int size, int initialUB)
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
};


#endif
