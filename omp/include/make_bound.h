#ifndef MAKE_BOUND_H_
#define MAKE_BOUND_H

#include "libbounds.h"

template<typename T>
std::unique_ptr<bound_abstract<int>> make_bound_ptr(instance_abstract inst, const int _bound_choice = 0)
{
    std::cout<<"make_bound_ptr\n"<<std::endl;

    if(arguments::problem[0]=='f'){
        switch (_bound_choice) {
            case 0:
            {
                auto bd = std::make_unique<bound_fsp_weak>();
                #pragma omp critical
                bd->init(inst);
                return bd;
            }
            case 1:
            {
                auto bd = std::make_unique<bound_fsp_strong>();
                #pragma omp critical
                bd->init(inst);
                bd->earlyExit=arguments::earlyStopJohnson;
                bd->machinePairs=arguments::johnsonPairs;
                return bd;
            }
        }
    }else if(arguments::problem[0]=='d'){
        std::cout<<"Dummy bound\n";
        auto bd = std::make_unique<bound_dummy>();
        bd->init(inst);
        return bd;
    }
    return nullptr;
}


#endif
