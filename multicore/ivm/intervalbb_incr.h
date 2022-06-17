#ifndef INTERVALBB_INCR_H_
#define INTERVALBB_INCR_H_

#include "intervalbb.h"

#include "libbounds.h"

// class pbab;

template<typename T>
class IntervalbbIncr : public Intervalbb<T>{
public:
    IntervalbbIncr(pbab* _pbb) : Intervalbb<T>(_pbb)
    {
        secondary_bound = this->pbb->bound_factory->make_bound(_pbb->instance,1);
    };


    void boundAndKeepSurvivors(subproblem& _subpb,const int mode) override
    {
        std::vector<std::vector<T>> lb(2,std::vector<T>(this->size,0));
        std::vector<std::vector<T>> prio(2,std::vector<T>(this->size,0));

        int dir = this->branch->pre_bound_choice(this->IVM->getDepth());

        if(dir<0){
            //get bounds for both children sets using incremental evaluator
            this->primary_bound->boundChildren(
                    _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                    lb[Branching::Front].data(),lb[Branching::Back].data(),
                    prio[Branching::Front].data(),prio[Branching::Back].data()
                );
            //choose branching direction
            dir = (*(this->branch))(
                lb[Branching::Front].data(),
                lb[Branching::Back].data(),
                this->IVM->getDepth()
            );
        }else if(dir==Branching::Front){
            this->primary_bound->boundChildren(
                    _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                    lb[Branching::Front].data(),nullptr,
                    prio[Branching::Front].data(),nullptr
                );
        }else if(dir==Branching::Back){
            this->primary_bound->boundChildren(
                    _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                    nullptr,lb[Branching::Back].data(),
                    nullptr,prio[Branching::Back].data()
                );
        }

        //branching direction was selected
        this->IVM->setDirection(dir);

        //now refine bounds
        //mark all subproblems not pruned by first bound
        std::vector<bool>mask(this->size,false);
        for (int i = _subpb.limit1 + 1; i < _subpb.limit2; i++) {
            int job = _subpb.schedule[i];
            if(!(*this->prune)(lb[dir][job])){
                mask[job] = true;
            }
        }
        if(dir==Branching::Front){
            int costs[2];
            for (int i = _subpb.limit1 + 1; i < _subpb.limit2; i++) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                this->secondary_bound->bornes_calculer(_subpb.schedule.data(), _subpb.limit1 + 1, _subpb.limit2, costs, this->prune->local_best);
                    lb[Branching::Front][job] = costs[0];
                    prio[Branching::Front][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                }
            }
        }else{
            int costs[2];
            for (int i = _subpb.limit2 - 1; i > _subpb.limit1; i--) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                this->secondary_bound->bornes_calculer(_subpb.schedule.data(), _subpb.limit1, _subpb.limit2-1, costs, this->prune->local_best);
                    lb[Branching::Back][job] = costs[0];
                    prio[Branching::Back][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                }
            }
        }

        this->IVM->sortSiblingNodes(
            lb[dir],
            prio[dir]
        );
        this->eliminateJobs(lb[dir]);
    }

    std::unique_ptr<bound_abstract<T>> secondary_bound ;
};

#endif
