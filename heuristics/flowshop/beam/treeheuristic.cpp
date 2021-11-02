#include "treeheuristic.h"
#include "operator_factory.h"

Treeheuristic::Treeheuristic(instance_abstract* inst)
{
    tr = std::make_unique<Tree>(inst,inst->size);
    tr->strategy = PRIOQ;

    prune = OperatorFactory::createPruning(arguments::findAll);
    branch= OperatorFactory::createBranching(arguments::branchingMode,inst->size,99999);

    eval = std::make_unique<bound_fsp_weak_idle>( );
    eval->init(inst);

    bestSolution = std::make_unique<subproblem>(inst->size);

    ls = std::make_unique<LocalSearch>(inst);
    ig = std::make_unique<IG>(inst);
    beam = std::make_unique<Beam>(inst);

}

int
Treeheuristic::run(subproblem *s, int _ub)
{
    // ig->shuffle(s->schedule.data(), s->size);

    subproblem *bsol=new subproblem(*s);
    bsol->ub = eval->evalSolution(bsol->schedule.data());

	prune->local_best=_ub;//ub used for pruning
    if(prune->local_best == 0)
        prune->local_best = bsol->ub; //if no upper bound provided

    subproblem *tmpsol = new subproblem(*bsol);
    subproblem *currsol = new subproblem(*bsol);

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

		if(tmpsol->ub < currsol->ub){
            *currsol = *tmpsol;
        }
        if(tmpsol->ub < bsol->ub){
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
Treeheuristic::exploreNeighborhood(subproblem* s,long long int cutoff)
{
    tr->clearPool();
    tr->setRoot(s->schedule, s->limit1, s->limit2);
    (tr->top())->cost = 0;
    (tr->top())->ub = eval->evalSolution(tr->top()->schedule.data());

    bool foundSolution = false;

    while(true)
    {
        if (!tr->empty()) {
            subproblem *n = tr->take();

            if(!(*prune)(n)){
                if(n->leaf()){
                    prune->local_best = n->cost;
                    *bestSolution = *n;
                    foundSolution = true;
                }else{
                    std::vector<subproblem*>ns;
                    ns = decompose(*n);

                    float alpha = (float)(n->depth)/n->size;
                    for(auto &c : ns){
                        c->prio = (1.0f-alpha)*(c->cost)*c->prio + alpha*(c->cost);
                    }

                    insert(ns);

                    if(tr->top()->ub < bestSolution->ub)
                    {
                        prune->local_best = tr->top()->ub;
                        *bestSolution = *(tr->top());
                        foundSolution = true;

                        std::cout<<"new th  "<<*bestSolution<<"\n";
                    }
                }
            }
            delete n;
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

std::vector<subproblem*>
Treeheuristic::decompose(subproblem& n)
{
    //offspring nodes
    std::vector<subproblem *>children;

    //temporary used in evaluation
    subproblem * tmp;

    if (n.simple()) { //2 solutions ...
        tmp        = new subproblem(n, n.limit1 + 1, BEGIN_ORDER);
        tmp->cost=eval->evalSolution(tmp->schedule.data());
        children.push_back(tmp);

        tmp        = new subproblem(n, n.limit1+2 , BEGIN_ORDER);
        tmp->cost=eval->evalSolution(tmp->schedule.data());
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
                    tmp = new subproblem(n, j, BEGIN_ORDER);

                    tmp->cost=costFwd[job];
                    tmp->prio=prioFwd[job];

                    children.push_back(tmp);
                }
            }
        }else{
            for (int j = n.limit2 - 1; j > n.limit1; j--) {
                int job = n.schedule[j];
                if(!(*prune)(costBwd[job])){
                    tmp = new subproblem(n, j, END_ORDER);

                    tmp->cost=costBwd[job];
                    tmp->prio=prioFwd[job];

                    children.push_back(tmp);
                }
            }
        }
    }
    return children;
}


void
Treeheuristic::insert(std::vector<subproblem *>&ns)
{
    //no children (decomposition avoid generation of unpromising children)
    if (!ns.size())
        return;

    //children inserted with push_back [ 1 2 3 ... ]
    //for left->right exploration, insert (push) in reverse order
    for (auto i = ns.rbegin(); i != ns.rend(); i++) {
        (*i)->ub = eval->evalSolution((*i)->schedule.data());
        tr->push(std::move(*i));
    }
}
