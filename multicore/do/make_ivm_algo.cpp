#include <arguments.h>
#include <pbab.h>

#include "make_ivm_algo.h"

template<typename T>
std::shared_ptr<Intervalbb<T>> make_ivmbb(pbab* pbb)
{
    std::shared_ptr<Intervalbb<T>> bb;

    if(arguments::boundMode == 0){
        bb = std::make_shared<Intervalbb<T>>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
    }else if(arguments::boundMode == 1){
        bb = std::make_shared<IntervalbbEasy<T>>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
    }else{
        bb = std::make_shared<IntervalbbIncr<T>>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
        bb->set_secondary_bound( make_bound_ptr<int>(pbb,arguments::secondary_bound));
    }

    bb->set_prune( make_prune_ptr<int>(pbb) );
    bb->set_branch( make_branch_ptr<int>(pbb) );

    return bb;
}

template std::shared_ptr<Intervalbb<int>> make_ivmbb<int>(pbab* pbb);
