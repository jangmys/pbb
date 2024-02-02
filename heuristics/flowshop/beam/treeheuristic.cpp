#include "treeheuristic.h"
#include "set_operators.h"

#include "pruning.h"
#include "branching.h"

Treeheuristic::Treeheuristic(pbab* _pbb,instance_abstract& inst) :
    pbb(_pbb),
    tr(std::make_unique<Tree>(inst,inst.size)),
    eval(std::make_unique<bound_fsp_weak_idle>())
{
    prune = make_prune_ptr<int>(_pbb->best_found.initial_cost);
    branch = make_branch_ptr<int>(_pbb->size,_pbb->best_found.initial_cost);

    tr->strategy = PRIOQ;

    eval->init(inst);
    bestSolution = std::make_unique<subproblem>(inst.size);

    ls = std::make_unique<LocalSearch>(inst);
    ig = std::make_unique<IG>(inst);
    beam = std::make_unique<Beam>(pbb,inst);
}


int
Treeheuristic::run(std::shared_ptr<subproblem>& s, int _ub)
{
    std::shared_ptr<subproblem> bsol = std::make_shared<subproblem>(*s);

    beam->run_loop(1<<10,bsol.get());
    *bsol = *(beam->bestSolution);

    beam->run_loop(1<<12,bsol.get());
    *bsol = *(beam->bestSolution);

	prune->local_best=_ub;//ub used for pruning
    if(prune->local_best == 0)
        prune->local_best = bsol->ub; //if no upper bound provided

    std::shared_ptr<subproblem> tmpsol = std::make_shared<subproblem>(*bsol);
    std::shared_ptr<subproblem> currsol = std::make_shared<subproblem>(*bsol);

    int c=0;
    long long int cutoff = bsol->size*bsol->size*10;
    bool perturb = false;

    while(1){
        if(perturb){
            int l = intRand(2, tmpsol->size/5);
            int r = intRand(0, tmpsol->size - l);

            std::random_device rd;
            std::mt19937 g(rd());

            std::shuffle(tmpsol->schedule.begin()+r, tmpsol->schedule.begin()+r+l, g);

            (tmpsol)->ub = (*ls)(tmpsol->schedule,-1,tmpsol->size);

			perturb=false;
		}

        tmpsol->limit1=-1;
		tmpsol->limit2=s->size;

        exploreNeighborhood(tmpsol,cutoff);

		if(tmpsol->ub < currsol->ub){
            *currsol = *tmpsol;
        }
        if(tmpsol->ub < bsol->ub){
            std::cout<<"improved "<<tmpsol->ub<<std::endl;
            *bsol = *tmpsol;
        }else{
            *tmpsol = *currsol;
            perturb = true;
            c++;
        }

        if(c > 500)
            break;
    }

    tr->clearPool();

    *s = *bsol;

    return 0;
}

void
Treeheuristic::exploreNeighborhood(std::shared_ptr<subproblem> s,long long int cutoff)
{
    tr->clearPool();
    tr->setRoot(s->schedule, s->limit1, s->limit2);
    (tr->top())->lb = 0;
    (tr->top())->ub = eval->evalSolution(tr->top()->schedule);

    bool foundSolution = false;

    while(true)
    {
        if (!tr->empty()) {
            std::shared_ptr<subproblem> n = tr->take();

            if(!(*prune)(n.get())){
                if(n->leaf()){
                    prune->local_best = n->lb;
                    *bestSolution = *n;
                    foundSolution = true;
                }else{
                    std::vector<std::shared_ptr<subproblem>>ns;
                    ns = decompose(*n);

                    float alpha = (float)(n->depth+1)/n->size;
                    for(auto &c : ns){
                        c->prio = (1.0f-alpha)*(c->lb)*c->prio + alpha*(c->lb);
                    }

                    insert(ns);

                    if(tr->top()->ub < bestSolution->ub)
                    {
                        prune->local_best = tr->top()->ub;
                        *bestSolution = *(tr->top());
                        foundSolution = true;
                        // std::cout<<"new treeheuristic best\t"<<*bestSolution<<"\n";
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

    if (n.is_simple()) { //2 solutions ...
        tmp        = std::make_shared<subproblem>(n, n.limit1 + 1, BEGIN_ORDER);
        tmp->lb = eval->evalSolution(tmp->schedule);
        children.push_back(tmp);

        tmp        = std::make_shared<subproblem>(n, n.limit1+2 , BEGIN_ORDER);
        tmp->lb = eval->evalSolution(tmp->schedule);
        children.push_back(tmp);
    } else {
        std::vector<int> costFwd(n.size);
        std::vector<int> costBwd(n.size);

        std::vector<float> prioFwd(n.size);
        std::vector<float> prioBwd(n.size);

        //evaluate lower bounds and priority
        eval->boundChildren(n.schedule,n.limit1,n.limit2, costFwd.data(),costBwd.data(), prioFwd.data(),prioBwd.data());
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

                    tmp->lb = costFwd[job];
                    tmp->prio=prioFwd[job];

                    children.push_back(tmp);
                }
            }
        }else{
            for (int j = n.limit2 - 1; j > n.limit1; j--) {
                int job = n.schedule[j];
                if(!(*prune)(costBwd[job])){
                    tmp = std::make_shared<subproblem>(n, j, END_ORDER);

                    tmp->lb = costBwd[job];
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
        if((*i)->depth < 50 && (*i)->depth%5 == 0){
            int nb_iter = 20;
            int f = ig->runIG((*i).get(),(*i)->limit1+1,(*i)->limit2, nb_iter);
            (*i)->ub = f;
        }else if((*i)->depth%5 == 0){
            int c = (*ls)((*i)->schedule,(*i)->limit1+1,(*i)->limit2);
            (*i)->ub = c;
        }else{
            (*i)->ub = eval->evalSolution((*i)->schedule);
        }

        tr->push(std::move(*i));
    }
}
