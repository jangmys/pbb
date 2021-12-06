#include "treeheuristic.h"
#include "operator_factory.h"


Treeheuristic::Treeheuristic(instance_abstract* inst) :
    tr(std::make_unique<Tree>(inst,inst->size)),
    prune(OperatorFactory::createPruning(arguments::findAll)),
    branch(OperatorFactory::createBranching(arguments::branchingMode,inst->size,99999)),
    eval(std::make_unique<bound_fsp_weak_idle>())
{
    tr->strategy = PRIOQ;

    eval->init(inst);
    bestSolution = std::make_unique<subproblem>(inst->size);

    ls = std::make_unique<LocalSearch>(inst);
    ig = std::make_unique<IG>(inst);
    beam = std::make_unique<Beam>(inst);
}


int
Treeheuristic::run(std::shared_ptr<subproblem>& s, int _ub)
{
    std::shared_ptr<subproblem> bsol = std::make_shared<subproblem>(*s);
    bsol->set_fitness(eval->evalSolution(bsol->schedule.data()));

	prune->local_best=_ub;//ub used for pruning
    if(prune->local_best == 0)
        prune->local_best = bsol->fitness(); //if no upper bound provided

    std::shared_ptr<subproblem> tmpsol = std::make_shared<subproblem>(*bsol);
    std::shared_ptr<subproblem> currsol = std::make_shared<subproblem>(*bsol);

    int c=0;
    long long int cutoff = 100000;
    bool perturb = false;

    while(1){
        if(perturb){
            int l = helper::intRand(2, tmpsol->size/5);
            int r = helper::intRand(0, tmpsol->size - l);

            std::random_device rd;
            std::mt19937 g(rd());

            std::shuffle(tmpsol->schedule.begin()+r, tmpsol->schedule.begin()+r+l, g);
			perturb=false;
		}

        tmpsol->limit1=-1;
		tmpsol->limit2=s->size;

        exploreNeighborhood(tmpsol,cutoff);

		if(tmpsol->fitness() < currsol->fitness()){
            *currsol = *tmpsol;
        }
        if(tmpsol->fitness() < bsol->fitness()){
            // std::cout<<"improved "<<tmpsol->ub<<std::endl;
            *bsol = *tmpsol;
        }else{
            *tmpsol = *currsol;
            perturb = true;
            c++;
        }

        if(c > 100)
            break;
    }

    tr->clearPool();

    *s = *bsol;
}

void
Treeheuristic::exploreNeighborhood(std::shared_ptr<subproblem> s,long long int cutoff)
{
    tr->clearPool();
    tr->setRoot(s->schedule, s->limit1, s->limit2);
    (tr->top())->set_lower_bound(0);
    (tr->top())->set_fitness(eval->evalSolution(tr->top()->schedule.data()));

    bool foundSolution = false;

    while(true)
    {
        if (!tr->empty()) {
            std::shared_ptr<subproblem> n = tr->take();

            if(!(*prune)(n.get())){
                if(n->leaf()){
                    prune->local_best = n->lower_bound();
                    *bestSolution = *n;
                    foundSolution = true;
                }else{
                    std::vector<std::shared_ptr<subproblem>>ns;
                    ns = decompose(*n);

                    // float alpha = (float)(n->depth)/n->size;
                    // for(auto &c : ns){
                    //     c->prio = (1.0f-alpha)*(c->cost)*c->prio + alpha*(c->cost);
                    // }

                    insert(ns);

                    if(tr->top()->fitness() < bestSolution->fitness())
                    {
                        prune->local_best = tr->top()->fitness();
                        *bestSolution = *(tr->top());
                        foundSolution = true;

                        std::cout<<"new th  "<<*bestSolution<<"\n";
                    }
                }
            }
        }

        if(tr->empty()){
            break;
        }
        if(foundSolution){
            break;
        }
        if(tr->size() > cutoff){
            break;
        }
    }

    *s = *bestSolution;
}

std::vector<std::shared_ptr<subproblem>>
Treeheuristic::decompose(subproblem& n)
{
    //offspring nodes
    std::vector<std::shared_ptr<subproblem>>children;

    //temporary used in evaluation
    std::shared_ptr<subproblem> tmp;

    if (n.simple()) { //2 solutions ...
        tmp        = std::make_shared<subproblem>(n, n.limit1 + 1, BEGIN_ORDER);
        tmp->set_lower_bound(eval->evalSolution(tmp->schedule.data()));
        children.push_back(tmp);

        tmp        = std::make_shared<subproblem>(n, n.limit1+2 , BEGIN_ORDER);
        tmp->set_lower_bound(eval->evalSolution(tmp->schedule.data()));
        children.push_back(tmp);
    } else {
        std::vector<int> costFwd(n.size);
        std::vector<int> costBwd(n.size);

        std::vector<float> prioFwd(n.size);
        std::vector<float> prioBwd(n.size);

        //evaluate lower bounds and priority
        eval->boundChildren(n.schedule.data(),n.limit1,n.limit2, costFwd.data(),costBwd.data(), prioFwd.data(),prioBwd.data());
        //branching heuristic
        int dir = (*branch)(
            costFwd.data(),costBwd.data(),n.depth
        );

        //generate children nodes
        if(dir==BEGIN_ORDER){
            for (int j = n.limit1 + 1; j < n.limit2; j++) {
                int job = n.schedule[j];

                if(!(*prune)(costFwd[job])){
                    tmp = std::make_shared<subproblem>(n, j, BEGIN_ORDER);

                    tmp->set_lower_bound(costFwd[job]);
                    tmp->prio=prioFwd[job];

                    children.push_back(tmp);
                }
            }
        }else{
            for (int j = n.limit2 - 1; j > n.limit1; j--) {
                int job = n.schedule[j];
                if(!(*prune)(costBwd[job])){
                    tmp = std::make_shared<subproblem>(n, j, END_ORDER);

                    tmp->set_lower_bound(costBwd[job]);
                    tmp->prio=prioFwd[job];

                    children.push_back(tmp);
                }
            }
        }
    }
    return children;
}


void
Treeheuristic::insert(std::vector<std::shared_ptr<subproblem>>&ns)
{
    //no children (decomposition avoid generation of unpromising children)
    if (!ns.size())
        return;

    //children inserted with push_back [ 1 2 3 ... ]
    //for left->right exploration, insert (push) in reverse order
    for (auto i = ns.rbegin(); i != ns.rend(); i++) {
        (*i)->set_fitness(eval->evalSolution((*i)->schedule.data()));
        tr->push(std::move(*i));
    }
}
