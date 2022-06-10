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

        int dir = this->branch->pre_bound_choice(this->IVM->getDepth());

        if(dir<0){
            //no branching decision : evaluate both sets
            std::vector<bool> mask(this->size,true);
            this->eval->get_children_bounds_full(
                _subpb,
                mask, _subpb.limit1 + 1,
                lb[Branching::Front],
                prio[Branching::Front],
                -1, Evaluator<T>::Secondary
            );
            this->eval->get_children_bounds_full(
                _subpb,
                mask, _subpb.limit2 - 1,
                lb[Branching::Back],
                prio[Branching::Back],
                -1, Evaluator<T>::Secondary
            );

            dir = (*(this->branch))(
                lb[Branching::Front].data(),
                lb[Branching::Back].data(),
                this->IVM->getDepth()
            );
        }else if(dir==Branching::Front){
            std::vector<bool> mask(this->size,true);
            this->eval->get_children_bounds_full(
                _subpb,
                mask, _subpb.limit1 + 1,
                lb[Branching::Front],
                prio[Branching::Front],
                -1, Evaluator<T>::Secondary
            );
        }else{
            std::vector<bool> mask(this->size,true);
            this->eval->get_children_bounds_full(
                _subpb,
                mask, _subpb.limit2 - 1,
                lb[Branching::Back],
                prio[Branching::Back],
                -1, Evaluator<T>::Secondary
            );
        }

        this->IVM->setDirection(dir);

        //all
        this->IVM->sortSiblingNodes(
            lb[dir],
            prio[dir]
        );
        this->eliminateJobs(lb[dir]);
    }
};

#endif
