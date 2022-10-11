#ifndef INTERVALBB_EASY_H_
#define INTERVALBB_EASY_H_

#include "intervalbb.h"

#include "libbounds.h"

// class pbab;

template<typename T>
class IntervalbbEasy : public Intervalbb<T>{
public:
    IntervalbbEasy(pbab* _pbb) : Intervalbb<T>(_pbb){};


    void boundAndKeepSurvivors(subproblem& _subpb,const int mode) override
    {
        std::vector<std::vector<T>> lb(2,std::vector<T>(this->size,0));
        std::vector<std::vector<T>> prio(2,std::vector<T>(this->size,0));

        //a priori choice of branching direction
        int dir = this->branch->pre_bound_choice(this->IVM->getDepth());
        std::vector<bool> mask(this->size,true);

        if(dir<0){
            //evaluate both children sets using full LB evalution function
            int costs[2];

            for (int i = _subpb.limit1 + 1; i < _subpb.limit2; i++) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    //front
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                    this->primary_bound->bornes_calculer(_subpb.schedule.data(), _subpb.limit1 + 1, _subpb.limit2, costs, this->prune->local_best);
                    lb[Branching::Front][job] = costs[0];
                    prio[Branching::Front][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                    //back
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                    this->primary_bound->bornes_calculer(_subpb.schedule.data(), _subpb.limit1, _subpb.limit2 - 1, costs, this->prune->local_best);
                    lb[Branching::Back][job] = costs[0];
                    prio[Branching::Back][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                }
            }

            //make Branching decision
            dir = (*this->branch)(
                lb[Branching::Front].data(),
                lb[Branching::Back].data(),
                this->IVM->getDepth()
            );
            // this->IVM->setDirection(dir);
        }else if(dir==Branching::Front){
            int costs[2];
            for (int i = _subpb.limit1 + 1; i < _subpb.limit2; i++) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                    this->primary_bound->bornes_calculer(_subpb.schedule.data(), _subpb.limit1 + 1, _subpb.limit2, costs, this->prune->local_best);
                    lb[Branching::Front][job] = costs[0];
                    prio[Branching::Front][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                }
            }
            // this->IVM->setDirection(dir);
        }else{
            int costs[2];
            for (int i = _subpb.limit2 - 1; i > _subpb.limit1; i--) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                    this->primary_bound->bornes_calculer(_subpb.schedule.data(), _subpb.limit1, _subpb.limit2-1, costs, this->prune->local_best);
                    lb[Branching::Back][job] = costs[0];
                    prio[Branching::Back][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                }
            }
        }
        this->IVM->setDirection(dir);

        this->IVM->sortSiblingNodes(
            lb[dir],
            prio[dir]
        );
        this->eliminateJobs(lb[dir]);
    }
};

#endif
