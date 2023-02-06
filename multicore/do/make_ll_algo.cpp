#include <arguments.h>
#include <pbab.h>

#include "make_ll_algo.h"
#include "set_operators.h"


std::shared_ptr<Poolbb> make_poolbb(pbab* pbb)
{
    std::shared_ptr<Poolbb> bb;

    if(arguments::boundMode == 0){
        bb = std::make_shared<Poolbb>(pbb);
        bb->set_bound( make_bound_ptr<int>(pbb,arguments::primary_bound), 0);
    }else if(arguments::boundMode == 1){
        bb = std::make_shared<PoolbbEasy>(pbb);
        bb->set_bound( make_bound_ptr<int>(pbb,arguments::primary_bound), 0);
    }else{
        bb = std::make_shared<PoolbbIncremental>(pbb);
        bb->set_bound( make_bound_ptr<int>(pbb,arguments::primary_bound), 0);
        bb->set_bound( make_bound_ptr<int>(pbb,arguments::secondary_bound), 1);
    }

    bb->set_prune( make_prune_ptr<int>(pbb) );
    bb->set_branch( make_branch_ptr<int>(pbb) );

    return bb;
}
