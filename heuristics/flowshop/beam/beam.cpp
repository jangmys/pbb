#include <algorithm>

#include "omp.h"
#include "beam.h"
#include "set_operators.h"

Beam::Beam(pbab* _pbb, instance_abstract& inst) :
    pbb(_pbb),
    tr(std::make_unique<Tree>(inst,inst.size)),
    eval(std::make_unique<bound_fsp_weak_idle>())
{
    prune = make_prune_ptr<int>(_pbb);
    branch = make_branch_ptr<int>(_pbb);

    tr->strategy = DEQUE;
    eval->init(inst);

    bestSolution = std::make_unique<subproblem>(inst.size);
}

int
Beam::run(const int maxBeamWidth, subproblem* p)
{
    prune->local_best = 999999;

    int beamWidth = 1;
    do{
        tr->setRoot(p->schedule,p->limit1,p->limit2);
        beamWidth *=2;
    }while(beamWidth < maxBeamWidth);

    *p = *bestSolution;

    return 0;//p->cost;
 }

 int
 Beam::run_loop(const int maxBeamWidth, subproblem* p)
 {
     int localBest = p->fitness();

     prune->local_best = p->fitness();

     int beamWidth = 1;
     do{
        activeSet.clear();

        std::unique_ptr<subproblem> root = std::make_unique<subproblem>(*p);
        root->set_lower_bound(0);
        activeSet.push_back(std::move(root));

        // while(step_loop(beamWidth,localBest));
        // while(step_loop_pq(beamWidth,localBest));
        while(step_loop_local_pq(beamWidth,localBest));

        beamWidth *=2;
     }while(beamWidth < maxBeamWidth);

     *p = *bestSolution;

     return 0;
}

bool
Beam::step_loop_local_pq(unsigned int beamWidth,int localBest){
    int n_parents = activeSet.size();

    if(n_parents>0 && activeSet[0]->depth == bestSolution->size-1){
        std::unique_ptr<subproblem> slice_min(std::move(activeSet[0]));
        for(int i=1;i<n_parents;i++){
            if(activeSet[i]->lower_bound() < slice_min->lower_bound()){
                slice_min = std::move(activeSet[i]);
            }
        }
        activeSet.clear();

        if(slice_min->lower_bound() < prune->local_best){
            slice_min->set_fitness(slice_min->lower_bound());
            prune->local_best = slice_min->fitness();
            bestSolution = std::move(slice_min);
            std::cout<<"best\t"<<*bestSolution<<"\n";
        }

        return false;
    }

    std::priority_queue<
        std::unique_ptr<subproblem>,
        std::vector<std::unique_ptr<subproblem>>,
        prio_compare>pq;

         // auto(*)(std::unique_ptr<subproblem>,std::unique_ptr<subproblem>)->bool > pq{
         //    []( const std::unique_ptr<subproblem> a, const std::unique_ptr<subproblem> b )->bool {
         //        return a->prio < b->prio;
         //    }
         //    };

    if(n_parents>0 && activeSet[0]->depth < bestSolution->size-1){
        #pragma omp parallel
        {
            std::priority_queue<
                std::unique_ptr<subproblem>,
                std::vector<std::unique_ptr<subproblem>>,
                prio_compare>local_pq;

            #pragma omp for schedule(static) nowait
            for(int i=0;i<n_parents;i++){
                std::unique_ptr<subproblem> n(std::move(activeSet[i]));
                std::vector<std::unique_ptr<subproblem>>ns;
                decompose(*n,ns);

                //compute priorities
                float alpha = (float)(n->depth+1)/n->size;
                for(auto &c : ns){
                    c->prio = (1.0f-alpha)*(c->lower_bound())*c->prio + alpha*(c->lower_bound());
                }
                for(auto it=std::make_move_iterator(ns.begin());
                    it!=std::make_move_iterator(ns.end());it++){
                    if(local_pq.size()<beamWidth){
                        local_pq.push(std::move(*it));
                    }else if((*it)->prio < local_pq.top()->prio){
                        local_pq.pop();
                        local_pq.push(std::move(*it));
                    }
                }
            }

            //merging pqueues
            #pragma omp critical
            {
                while(!local_pq.empty()) {
                    if(pq.size()<beamWidth){
                        pq.push(std::move(const_cast<std::unique_ptr<subproblem>&>(local_pq.top())));
                    }else if(local_pq.top()->prio < pq.top()->prio){
                        pq.pop();
                        pq.push(std::move(const_cast<std::unique_ptr<subproblem>&>(local_pq.top())));
                    }
                    local_pq.pop();
                }
            }
        }
    }

    if(pq.empty()){
        return false;
    }

    activeSet.clear();

    while(!pq.empty()) {
        activeSet.push_back(
            std::move(const_cast<std::unique_ptr<subproblem>&>(pq.top()))
        );
        pq.pop();
    }
//
    return true;
}




bool
Beam::step_loop_pq(unsigned int beamWidth,int localBest){
    int n_parents = activeSet.size();

    if(n_parents>0 && activeSet[0]->depth == bestSolution->size-1){
        std::unique_ptr<subproblem> slice_min(std::move(activeSet[0]));
        for(int i=1;i<n_parents;i++){
            if(activeSet[i]->lower_bound() < slice_min->lower_bound()){
                slice_min = std::move(activeSet[i]);
            }
        }
        activeSet.clear();

        if(slice_min->lower_bound() < prune->local_best){
            slice_min->set_fitness(slice_min->lower_bound());
            prune->local_best = slice_min->fitness();
            bestSolution = std::move(slice_min);
            std::cout<<"best\t"<<*bestSolution<<"\n";
        }

        return false;
    }

    //largest (cutoff-value) is on top
    std::priority_queue<
        std::unique_ptr<subproblem>,
        std::vector<std::unique_ptr<subproblem>>,
        prio_compare>pq;

    struct timespec t1,t2;
    clock_gettime(CLOCK_MONOTONIC,&t1);

    if(n_parents>0 && activeSet[0]->depth < bestSolution->size-1){
        #pragma omp parallel
        {
            std::vector<std::unique_ptr<subproblem>>local_children;
            #pragma omp for schedule(static,100)
            for(int i=0;i<n_parents;i++){
                std::unique_ptr<subproblem> n(std::move(activeSet[i]));
                std::vector<std::unique_ptr<subproblem>>ns;
                decompose(*n,ns);

                //compute priorities
                float alpha = (float)(n->depth+1)/n->size;
                for(auto &c : ns){
                    c->prio = (1.0f-alpha)*(c->lower_bound())*c->prio + alpha*(c->lower_bound());
                }
                //insert in next slice
                local_children.insert(local_children.end(), make_move_iterator(ns.begin()), make_move_iterator(ns.end()));
            }


            #pragma omp critical
            {
                // std::cout<<"thread "<<omp_get_thread_num()<<" "<<local_children.size()<<"/"<<beamWidth<<"\n";
                for(auto &c: local_children){
                    if(pq.size()<beamWidth){
                        pq.push(std::move(c));
                    }else if(c->prio < pq.top()->prio){
                        pq.pop();
                        pq.push(std::move(c));
                    }
                }
                // std::cout<<"PQ SIZE\t"<<pq.size()<<"/"<<beamWidth<<"\n";
            }
        }
    }

    if(pq.empty()){
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC,&t2);
    activeSet.clear();

    while(!pq.empty()) {
        activeSet.push_back(
            std::move(const_cast<std::unique_ptr<subproblem>&>(pq.top()))
        );
        pq.pop();
    }

    return true;
}

bool
Beam::step_loop(unsigned int beamWidth,int localBest){
    int n_parents = activeSet.size();

    if(n_parents>0 && activeSet[0]->depth == bestSolution->size-1){
        std::unique_ptr<subproblem> slice_min(std::move(activeSet[0]));
        for(int i=1;i<n_parents;i++){
            if(activeSet[i]->lower_bound() < slice_min->lower_bound()){
                slice_min = std::move(activeSet[i]);
            }
        }
        activeSet.clear();

        if(slice_min->lower_bound() < prune->local_best){
            slice_min->set_fitness(slice_min->lower_bound());
            prune->local_best = slice_min->fitness();
            bestSolution = std::move(slice_min);
            std::cout<<"best\t"<<*bestSolution<<"\n";
        }

        return false;
    }

    //next slice
    std::vector<std::unique_ptr<subproblem>>children;

    struct timespec t1,t2;
    clock_gettime(CLOCK_MONOTONIC,&t1);

    if(n_parents>0 && activeSet[0]->depth < bestSolution->size-1){
        #pragma omp parallel
        {
            std::vector<std::unique_ptr<subproblem>>local_children;
            #pragma omp for schedule(static)
            for(int i=0;i<n_parents;i++){
                std::unique_ptr<subproblem> n(std::move(activeSet[i]));
                std::vector<std::unique_ptr<subproblem>>ns;
                decompose(*n,ns);

                //compute priorities
                float alpha = (float)(n->depth)/n->size;
                for(auto &c : ns){
                    c->prio = (1.0f-alpha)*(c->lower_bound())*c->prio + alpha*(c->lower_bound());
                }
                //insert in next slice
                local_children.insert(local_children.end(), make_move_iterator(ns.begin()), make_move_iterator(ns.end()));
            }

            #pragma omp critical
            {
                // std::cout<<"thread "<<omp_get_thread_num()<<" insert "<<ns.size()<<"\n";
                children.insert(children.end(), make_move_iterator(local_children.begin()), make_move_iterator(local_children.end()));
            }
        }
    }

    if(children.empty()){
        return false;
    }
    //
    clock_gettime(CLOCK_MONOTONIC,&t2);
    // std::cout<<"FillSlice\t"<<children.size()<<"/"<<beamWidth<<"\t"<<(t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec)/1e9<<"\t"<<std::flush;

    activeSet.clear();
    //
    // clock_gettime(CLOCK_MONOTONIC,&t1);
    //
    //sort slice (or make tr a pqueue insert directly with check on length...)
    std::sort(children.begin(),children.end(),
        [](const std::unique_ptr<subproblem>& a,const std::unique_ptr<subproblem>& b){
            return a->prio > b->prio;
        }
    );
    //
    // clock_gettime(CLOCK_MONOTONIC,&t2);
    // // std::cout<<"Sort\t"<<(t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec)/1e9<<"\t"<<std::flush;
    //
    // clock_gettime(CLOCK_MONOTONIC,&t1);
    //
    for (auto i = children.rbegin(); i != children.rend(); i++) {
        // delete (*i);
    	if(activeSet.size()<beamWidth){
            activeSet.push_back(std::move(*i));
	    }
	}
    //
    // clock_gettime(CLOCK_MONOTONIC,&t2);
    // // std::cout<<"Insert\t"<<(t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec)/1e9<<std::endl;
    //
    return true;
}


// bool
// Beam::step(int beamWidth,int localBest)
// {
//     //current state
//     std::shared_ptr<subproblem> n;
//     //next slice
//     std::vector<std::shared_ptr<subproblem>>children;
//
//     while (!tr->empty()) {
//         n = tr->take();
//         if(!(*prune)(n.get())){
//             if (n->leaf()) {
//                 n->ub = n->cost;
//                 prune->local_best = n->ub;
//                 *bestSolution = *n;
//             }else{
//                 //expand (compute lower bounds, priorities and generate surviving successors)
//                 std::vector<std::unique_ptr<subproblem>>ns;
//                 ns = decompose(*n);
//
//                 //compute priorities
//                 float alpha = (float)(n->depth)/n->size;
//                 for(auto &c : ns){
//                     c->prio = (1.0f-alpha)*(c->cost)*c->prio + alpha*(c->cost);
//                 }
//
//                 //insert in next slice
//                 children.insert(children.end(), make_move_iterator(ns.begin()), make_move_iterator(ns.end()));
//             }
//         }
// 	}
//
//     //sort slice (or make tr a pqueue insert directly with check on length...)
//     std::sort(children.begin(),children.end(),
//         [](std::shared_ptr<subproblem> a,std::shared_ptr<subproblem> b){
//             return a->prio > b->prio;
//         }
//     );
//
//     //truncate
//     for (auto i = children.rbegin(); i != children.rend(); i++) {
//     	if(tr->size()<beamWidth){
// 			tr->push(std::move(*i));
// 	    }
// 	}
//
//     return !tr->empty();
// }

void
Beam::decompose(const subproblem& n, std::vector<std::unique_ptr<subproblem>>& children)
{
    //offspring nodes
    // std::vector<std::unique_ptr<subproblem>>children;

    //temporary used in evaluation
    std::unique_ptr<subproblem> tmp;

    if (n.simple()) { //2 solutions ...
        tmp        = std::make_unique<subproblem>(n, n.limit1 + 1, BEGIN_ORDER);
        tmp->set_lower_bound(eval->evalSolution(tmp->schedule.data()));
        children.push_back(std::move(tmp));

        tmp        = std::make_unique<subproblem>(n, n.limit1+2 , BEGIN_ORDER);
        tmp->set_lower_bound(eval->evalSolution(tmp->schedule.data()));
        children.push_back(std::move(tmp));
    } else {
        std::vector<int> costFwd(n.size);
        std::vector<int> costBwd(n.size);

        std::vector<float> prioFwd(n.size);
        std::vector<float> prioBwd(n.size);

        //evaluate lower bounds and priority
        eval->boundChildren(n.schedule.data(),n.limit1,n.limit2, costFwd.data(),costBwd.data(), prioFwd.data(),prioBwd.data());
        //branching heuristic
        int dir = (*branch)(costFwd.data(),costBwd.data(),n.depth);

        //generate children nodes
        if(dir==BEGIN_ORDER){
            for (int j = n.limit1 + 1; j < n.limit2; j++) {
                int job = n.schedule[j];

                if(!(*prune)(costFwd[job])){
                    tmp = std::make_unique<subproblem>(n, j, BEGIN_ORDER);

                    tmp->set_lower_bound(costFwd[job]);
                    tmp->prio=prioFwd[job];

                    children.push_back(std::move(tmp));
                }
            }
        }else{
            for (int j = n.limit2 - 1; j > n.limit1; j--) {
                int job = n.schedule[j];
                if(!(*prune)(costBwd[job])){
                    tmp = std::make_unique<subproblem>(n, j, END_ORDER);

                    tmp->set_lower_bound(costBwd[job]);
                    tmp->prio=prioBwd[job];

                    children.push_back(std::move(tmp));
                }
            }
        }
    }
    // return children;
}
