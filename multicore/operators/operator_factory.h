#include "arguments.h"

#include "evaluator.h"
#include "branching.h"
#include "pruning.h"

#include "libbounds.h"

class OperatorFactory
{
  public:
    int a;

    static std::unique_ptr<pruning> createPruning(bool findAll)
    {
        if(!findAll)
        {
            return std::make_unique<keepSmaller>();
        }else{
            return std::make_unique<keepEqualOrSmaller>();
        }
    }

    static std::unique_ptr<branching> createBranching(int choice, int size, int initialUB)
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

    static std::unique_ptr<bound_abstract<int>> createBound(instance_abstract* instance, int nb)
    {
        if(arguments::problem[0]=='f'){
            if(arguments::boundMode == 0){
                if(nb==0){
                    auto bd = std::make_unique<bound_fsp_weak>( );

                    bd->init(instance);
                    return bd;
                }
                if(nb==1){
                    return nullptr;
                }
            }
            if(arguments::boundMode == 1){
                if(nb==0){
                    auto bd2 = std::make_unique<bound_fsp_strong>( );
                    bd2->init(instance);
                    bd2->earlyExit=arguments::earlyStopJohnson;
                    bd2->machinePairs=arguments::johnsonPairs;
                    return bd2;
                }
                if(nb==1){
                    auto bd2 = std::make_unique<bound_fsp_strong>( );
                    bd2->init(instance);
                    bd2->earlyExit=arguments::earlyStopJohnson;
                    bd2->machinePairs=arguments::johnsonPairs;
                    return bd2;
                    // return nullptr;
                }
            }
            if(arguments::boundMode == 2){
                if(nb==0){
                    auto bd = std::make_unique<bound_fsp_weak>();
                    bd->init(instance);
                    return bd;
                }
                if(nb==1){
                    auto bd2 = std::make_unique<bound_fsp_strong>();
                    bd2->init(instance);
                    bd2->branchingMode=arguments::branchingMode;
                    bd2->earlyExit=arguments::earlyStopJohnson;
                    bd2->machinePairs=arguments::johnsonPairs;
                    return bd2;
                }
            }

    	}

    	std::cout<<"CreateBound: unknown problem\n";
    	return nullptr;
    }

    static std::unique_ptr<evaluator<int>> createEvaluator(instance_abstract* instance, int nb)
    {
        switch (arguments::boundMode) {
            case 0:{
                auto ev = std::make_unique<evaluator<int>>(
                    std::make_unique<bound_fsp_weak>()
                );
                ev->lb->init(instance);
                return ev;
            }
            case 1:{
                auto bd2 = std::make_unique<bound_fsp_strong>( );
                bd2->init(instance);
                bd2->earlyExit=arguments::earlyStopJohnson;
                bd2->machinePairs=arguments::johnsonPairs;

                auto ev = std::make_unique<evaluator<int>>(
                    std::move(bd2)
                );
                return ev;
            }
            case 2:{
                auto bd = std::make_unique<bound_fsp_weak>();
                bd->init(instance);

                auto bd2 = std::make_unique<bound_fsp_strong>( );
                bd2->init(instance);
                bd2->earlyExit=arguments::earlyStopJohnson;
                bd2->machinePairs=arguments::johnsonPairs;

                auto ev = std::make_unique<evaluator<int>>(
                    std::move(bd),
                    std::move(bd2)
                );
                return ev;
            }

        }
    }
};
