#include <arguments.h>
#include <pbab.h>

#include "make_ll_algo.h"
#include "set_operators.h"


std::shared_ptr<Poolbb> make_poolbb(pbab* pbb)
{
    std::shared_ptr<Poolbb> bb;

    if(arguments::boundMode == 0){
        bb = std::make_shared<Poolbb>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
    }else if(arguments::boundMode == 1){
        bb = std::make_shared<PoolbbEasy>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
    }else{
        bb = std::make_shared<PoolbbIncremental>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
        bb->set_secondary_bound( make_bound_ptr<int>(pbb,arguments::secondary_bound));
    }

    bb->set_prune( make_prune_ptr<int>(pbb->best_found.initial_cost) );
    bb->set_branch( make_branch_ptr<int>(pbb->size,pbb->best_found.initial_cost) );

    return bb;
}
