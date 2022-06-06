#ifndef OPERATOR_FACTORY_H
#define OPERATOR_FACTORY_H

#include "arguments.h"

#include "evaluator.h"
#include "branching.h"
#include "pruning.h"

#include "libbounds.h"


class PruningFactoryInterface
{
public:
    virtual std::unique_ptr<Pruning> make_pruning() = 0;
};

class PruneStrictLargerFactory : public PruningFactoryInterface
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
    virtual std::unique_ptr<Branching> make_branching(int choice, int size, int initialUB) = 0;
};

class PFSPBranchingFactory : public BranchingFactoryInterface
{
public:
    std::unique_ptr<Branching> make_branching(int choice, int size, int initialUB) override
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


//===================================================================
//===================================================================
//===================================================================

template<typename T>
class BoundFactoryInterface
{
public:
    virtual std::unique_ptr<bound_abstract<T>> make_bound(std::unique_ptr<instance_abstract>& inst, int bound_mode) = 0;
};

template<typename T>
class PFSPBoundFactory : public BoundFactoryInterface<T>
{
public:
    std::unique_ptr<bound_abstract<T>> make_bound(std::unique_ptr<instance_abstract>& inst, int bound_mode) override
    {
        switch (bound_mode) {
            case 0:
            {
                std::unique_ptr<bound_fsp_weak> bd = std::make_unique<bound_fsp_weak>();
                bd->init(inst.get());
                return bd;
            }
            case 1:
            {
                std::unique_ptr<bound_fsp_strong> bd = std::make_unique<bound_fsp_strong>();
                bd->init(inst.get());
                bd->earlyExit=arguments::earlyStopJohnson;
                bd->machinePairs=arguments::johnsonPairs;
                return bd;
            }
        }
    }
};





class OperatorFactory
{
  public:
    // int a;

    // static std::unique_ptr<Pruning> createPruning(bool findAll)
    // {
    //     if(!findAll)
    //     {
    //         return std::make_unique<keepSmaller>();
    //     }else{
    //         return std::make_unique<keepEqualOrSmaller>();
    //     }
    // }

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


    static std::unique_ptr<Evaluator<int>> createEvaluator(std::unique_ptr<bound_abstract<int>> lb1, std::unique_ptr<bound_abstract<int>> lb2)
    {
        switch (arguments::problem[0]) {
            case 'f':
            {
                switch (arguments::boundMode) {
                    case 0:{
                        auto ev = std::make_unique<Evaluator<int>>(
                            std::move(lb1)
                        );
                        return ev;
                    }
                    case 1:{
                        auto ev = std::make_unique<Evaluator<int>>(
                            std::move(lb1),
                            std::move(lb2)
                        );
                        return ev;
                    }
                    case 2:{
                        auto ev = std::make_unique<Evaluator<int>>(
                            std::move(lb1),
                            std::move(lb2)
                        );
                        return ev;
                    }
                }
            }
            case 'd':
            {
                auto ev = std::make_unique<Evaluator<int>>(
                            std::make_unique<bound_dummy>()
                        );
                        // ev->lb->init(instance);
                return ev;
            }
        }
        return nullptr;
    }
};


#endif
