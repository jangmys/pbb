#include <arguments.h>
#include <pbab.h>

#include "make_ivm_algo.h"
#include "set_operators.h"

template<typename T>
std::shared_ptr<Intervalbb<T>> make_ivmbb(pbab* pbb)
{
    std::shared_ptr<Intervalbb<T>> bb;

    if(arguments::boundMode == 0){
        bb = std::make_shared<Intervalbb<T>>(pbb);
        bb->set_bound( make_bound_ptr<int>(pbb,arguments::primary_bound), 0);
    }else if(arguments::boundMode == 1){
        bb = std::make_shared<IntervalbbEasy<T>>(pbb);
        bb->set_bound( make_bound_ptr<int>(pbb,arguments::primary_bound), 0);
    }else{
        bb = std::make_shared<IntervalbbIncr<T>>(pbb);
        bb->set_bound( make_bound_ptr<int>(pbb,arguments::primary_bound), 0);
        bb->set_bound( make_bound_ptr<int>(pbb,arguments::secondary_bound), 1);
    }

    bb->set_prune( make_prune_ptr<int>(pbb) );
    bb->set_branch( make_branch_ptr<int>(pbb) );

    return bb;
}

template std::shared_ptr<Intervalbb<int>> make_ivmbb<int>(pbab* pbb);
