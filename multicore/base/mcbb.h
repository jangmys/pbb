/*base class for BB
*/
#ifndef MCBB_H_
#define MCBB_H_

#include <pbab.h>
#include <branching.h>
#include <pruning.h>

template<typename T>
class MCbb
{
public:
    MCbb(pbab* _pbb) : count_leaves(0),count_decomposed()
    {
        //default
        prune = std::make_unique<keepSmaller>(_pbb->best_found.initial_cost);
        branch = std::make_unique<minMinBranching>(_pbb->size,_pbb->best_found.initial_cost);
    }

    virtual ~MCbb(){};

    unsigned long long get_leaves_count() const
    {
        return count_leaves;
    }
    unsigned long long get_decomposed_count() const
    {
        return count_decomposed;
    }
    void reset_node_counter(){
        count_leaves = 0;
        count_decomposed = 0;
    }


    void set_branch(std::unique_ptr<Branching> _branch)
    {
        branch=std::move(_branch);
    }

    void set_prune(std::unique_ptr<Pruning> _prune)
    {
        prune=std::move(_prune);
    }

    void set_bound(std::unique_ptr<bound_abstract<T>> _bound, const int _bound_choice = 0)
    {
        if(_bound_choice == 0){
            primary_bound = std::move(_bound);
        }else{
            secondary_bound = std::move(_bound);
        }
    }

protected:
    unsigned long long count_leaves;
    unsigned long long count_decomposed;

    std::unique_ptr<Branching> branch;
    std::unique_ptr<Pruning> prune;
    std::unique_ptr<bound_abstract<T>> primary_bound ;
    std::unique_ptr<bound_abstract<T>> secondary_bound ;
};


#endif
