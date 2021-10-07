#include "beam.h"
#include "operator_factory.h"

Beam::Beam(instance_abstract* inst)
{
    tr = std::make_unique<Tree>(inst,inst->size);
    tr->strategy = DEQUE;

    prune = OperatorFactory::createPruning(arguments::findAll);
    branch= OperatorFactory::createBranching(arguments::branchingMode,inst->size,99999);
    lb.push_back(std::move(OperatorFactory::createBound(inst,0)));

    bestSolution = std::make_unique<solution>(inst->size);
}

int
Beam::run(const int maxBeamWidth, subproblem* p)
{
    int localBest = 999999;

    prune->local_best = 999999;


    int beamWidth = 1;
    do{
        tr->setRoot(p->schedule,p->limit1,p->limit2);

        while(step(beamWidth,localBest));

        beamWidth *=2;
    }while(beamWidth < maxBeamWidth);
 }

bool
Beam::step(int beamWidth,int localBest)
{
    //current state
    subproblem * n;
    //next slice
    std::vector<subproblem*>children;

    while (!tr->empty()) {
        n = tr->take();
        if(!(*prune)(n)){
            if (n->leaf()) {
                prune->local_best = n->cost;
                bestSolution->update(n->schedule.data(),n->cost);

                std::cout<<prune->local_best<<" === > \n";
            }else{
                //expand
                std::vector<subproblem*>ns;
                ns = decompose(*n);

                //compute priorities
                float alpha = (float)(n->depth)/n->size;
                for(auto &c : ns){
                    c->prio = (1.0f-alpha)*(c->cost)*c->prio + alpha*(c->cost);
                }

                //insert in next slice
                children.insert(children.end(), make_move_iterator(ns.begin()), make_move_iterator(ns.end()));
            }
        }
        delete(n);
	}

    //sort slice (or make it pqueue...)
    std::sort(children.begin(),children.end(),
        [](subproblem* a,subproblem* b){
            return a->prio > b->prio;
        }
    );

    //truncate
    for (auto i = children.rbegin(); i != children.rend(); i++) {
    	if(tr->size()<beamWidth){
			tr->push(std::move(*i));
	    }else{
            delete (*i);
    	}
	}

    return !tr->empty();
}

std::vector<subproblem*>
Beam::decompose(subproblem& n)
{
    //offspring nodes
    std::vector<subproblem *>children;

    //temporary used in evaluation
    subproblem * tmp;

    if (n.simple()) { //2 solutions ...
        tmp        = new subproblem(n, n.limit1 + 1, BEGIN_ORDER);
        tmp->cost=lb[SIMPLE]->evalSolution(tmp->schedule.data());
        children.push_back(tmp);

        tmp        = new subproblem(n, n.limit1+2 , BEGIN_ORDER);
        tmp->cost=lb[SIMPLE]->evalSolution(tmp->schedule.data());
        children.push_back(tmp);
    } else {
        std::vector<int> costFwd(n.size);
        std::vector<int> costBwd(n.size);

        std::vector<int> prioFwd(n.size);
        std::vector<int> prioBwd(n.size);

        //evaluate lower bounds and priority
        lb[SIMPLE]->boundChildren(n.schedule.data(),n.limit1,n.limit2,costFwd.data(),costBwd.data(),prioFwd.data(),prioBwd.data());
        //branching heuristic
        int dir = (*branch)(costFwd.data(),costBwd.data(),n.depth);

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
