#include "pbab.h"
#include "pool.h"

#include "poolbb.h"


Poolbb::Poolbb(pbab* _pbb) : pool(std::make_unique<Pool>(_pbb->size)),pbsize(_pbb->size) {
    prune = _pbb->pruning_factory->make_pruning();
    branch = _pbb->branching_factory->make_branching();
    bound = _pbb->bound_factory->make_bound(_pbb->instance,0);
};
// {};


void Poolbb::set_root(subproblem& node)
{
    std::cout<<"Poolbb : set_root\n";
}

void Poolbb::set_root(solution& node)
{
    pool->set_root(node.perm,-1,pbsize);
}

void Poolbb::run()
{
    std::cout<<"hello\n";

    unsigned long long int count = 0;

    while(1){
        // pbb->sltn->getBest(prune->local_best);

        if(!pool->empty()){
            auto n = std::move(pool->take());

            if(n->leaf()){
                if(!(*prune)(n->lower_bound()))
                {
                    prune->local_best = n->lower_bound();
                    std::cout<<n->lower_bound()<<"\t"<<prune->local_best<<"\n";
                }
                count++;
            }

            // std::cout<<"hello\t"<<*n<<"\n";

            std::vector<std::unique_ptr<subproblem>>ns;
            decompose(*(n.get()),ns);

            for(auto c = ns.rbegin(); c != ns.rend(); c++)
            {
                pool->push(std::move(*c));
            }

            // std::cout<<"hello\t"<<pool->size()<<"\n";
        }else{
            break;
        }
    }

    std::cout<<"nnodes\t"<<count<<"\n";
}

void
Poolbb::decompose(
    subproblem& n,
    // std::unique_ptr<subproblem> n,
    std::vector<std::unique_ptr<subproblem>>& children
){
    //temporary used in evaluation
    std::unique_ptr<subproblem> tmp;

    if (n.simple()) { //2 solutions ...
        tmp        = std::make_unique<subproblem>(n, n.limit1 + 1, Branching::Front);
        // tmp->set_lower_bound(0);
        tmp->set_lower_bound(bound->evalSolution(tmp->schedule.data()));
        children.push_back(std::move(tmp));

        tmp        = std::make_unique<subproblem>(n, n.limit1+2 , Branching::Front);
        // tmp->set_lower_bound(0);
        tmp->set_lower_bound(bound->evalSolution(tmp->schedule.data()));
        // std::cout<<tmp->fitness()<<" simple\n";
        children.push_back(std::move(tmp));
    } else {
        std::vector<int> costFwd(n.size);
        std::vector<int> costBwd(n.size);

        std::vector<int> prioFwd(n.size);
        std::vector<int> prioBwd(n.size);

        //evaluate lower bounds and priority
        // bound->boundChildren(n.int * permut, int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin,
          // int * prioEnd);
        bound->boundChildren(n.schedule.data(),n.limit1,n.limit2, costFwd.data(),costBwd.data(), prioFwd.data(),prioBwd.data());

        //branching heuristic
        int dir = (*branch)(costFwd.data(),costBwd.data(),n.depth);

        //generate children nodes
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
}
