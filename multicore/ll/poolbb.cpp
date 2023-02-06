#include "pbab.h"
#include "pool.h"

#include "poolbb.h"

Poolbb::Poolbb(pbab* _pbb) :
    MCbb<int>(_pbb),
    pool(std::make_unique<Pool>(_pbb->size)),
    pbsize(_pbb->size)
    {
        //default operators (see make_ll_algo.h for options)
        // prune = std::make_unique<keepSmaller>(_pbb->best_found.initial_cost);
        // branch = std::make_unique<minMinBranching>(pbsize,_pbb->best_found.initial_cost);
        // set_bound(
        //     _pbb->bound_factory->make_bound(_pbb->inst,0)
        // );
        // primary_bound = _pbb->bound_factory->make_bound(_pbb->inst,0);

        _pbb->best_found.getBest(prune->local_best);
        std::cout<<"initial UB\t"<<prune->local_best<<"\n";

        reset_node_counter();
    };


void Poolbb::set_root(subproblem& node)
{
    pool->insert(std::make_unique<subproblem>(node));
    std::cout<<"Poolbb : set_root\n";
}

void Poolbb::run()
{
    while(next());

    std::cout<<"nnodes\t"<<count_decomposed<<"\t"<<count_leaves<<"\n";
}

bool Poolbb::next()
{
    if(!pool->empty()){
        auto n = std::move(pool->take_top());

        if(n->leaf()){
            if(!(*prune)(n->lower_bound()))
            {
                prune->local_best = n->lower_bound();
                std::cout<<n->lower_bound()<<"\t"<<prune->local_best<<"\n";
            }
            count_leaves++;
        }

        std::vector<std::unique_ptr<subproblem>>ns = decompose(*(n.get()));

        count_decomposed++;

        for(auto c = ns.rbegin(); c != ns.rend(); c++)
        {
            pool->insert(std::move(*c));
        }
        return true;
    }else{
        return false;
    }
}

std::vector<std::unique_ptr<subproblem>>
Poolbb::decompose(subproblem& n){
    std::vector<std::unique_ptr<subproblem>>children;
    std::unique_ptr<subproblem> tmp;

    //if only 2 solutions ...
    if (n.simple()) {
        tmp = std::make_unique<subproblem>(n, n.limit1 + 1, Branching::Front);
        tmp->set_lower_bound(primary_bound->evalSolution(tmp->schedule.data()));
        if(!(*prune)(tmp.get()))
            children.push_back(std::move(tmp));

        tmp = std::make_unique<subproblem>(n, n.limit1+2 , Branching::Front);
        tmp->set_lower_bound(primary_bound->evalSolution(tmp->schedule.data()));
        if(!(*prune)(tmp.get()))
            children.push_back(std::move(tmp));
    } else {
        std::vector<int> costFwd(n.size);
        std::vector<int> costBwd(n.size);

        std::vector<int> prioFwd(n.size);
        std::vector<int> prioBwd(n.size);

        //a priori choice of branching direction
        auto dir = this->branch->pre_bound_choice(n.depth);

        if(dir<0){
            //eval begin-end
            this->primary_bound->boundChildren( n.schedule.data(),n.limit1,n.limit2, costFwd.data(),costBwd.data(),  prioFwd.data(),prioBwd.data(), this->prune->local_best
            );

            dir = (*branch)(costFwd.data(),costBwd.data(),n.depth);
        }else if(dir == Branching::Front){
            //only begin
            this->primary_bound->boundChildren( n.schedule.data(),n.limit1,n.limit2, costFwd.data(),nullptr,  prioFwd.data(),nullptr, this->prune->local_best
            );
        }else{
            //only end
            this->primary_bound->boundChildren( n.schedule.data(),n.limit1,n.limit2, nullptr,costBwd.data(), nullptr,prioBwd.data(), this->prune->local_best
            );
        }

        //generate children nodes and assign bounds
        if(dir==Branching::Front){
            for (int j = n.limit1 + 1; j < n.limit2; j++) {
                int job = n.schedule[j];

                if(!(*prune)(costFwd[job])){
                    tmp = std::make_unique<subproblem>(n, j, Branching::Front);
                    tmp->set_lower_bound(costFwd[job]);
                    tmp->prio=prioFwd[job];
                    children.push_back(std::move(tmp));
                }
            }
        }else{
            for (int j = n.limit2 - 1; j > n.limit1; j--) {
                int job = n.schedule[j];
                if(!(*prune)(costBwd[job])){
                    tmp = std::make_unique<subproblem>(n, j, Branching::Back);
                    tmp->set_lower_bound(costBwd[job]);
                    tmp->prio=prioBwd[job];
                    children.push_back(std::move(tmp));
                }
            }
        }
    }

    return children;
}
