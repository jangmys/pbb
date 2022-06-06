#ifndef INTERVALBB_EASY_H_
#define INTERVALBB_EASY_H_

#include "intervalbb.h"

#include "libbounds.h"

class pbab;

template<typename T>
class IntervalbbEasy : public Intervalbb<T>{
public:
    IntervalbbEasy(pbab* _pbb,
        std::unique_ptr<Branching> _branch,
        std::unique_ptr<Pruning> _prune
    ) : Intervalbb<T>(_pbb,std::move(_branch),std::move(_prune)){};


    void boundAndKeepSurvivors(subproblem& _subpb,const int mode) override
    {
        std::vector<std::vector<T>> lb(2,std::vector<T>(this->size,0));
        std::vector<std::vector<T>> prio(2,std::vector<T>(this->size,0));

        //weak or mixed bounding
        if(mode != 2){
            // for full evaluation
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
        }

        //make Branching decision
        auto dir = (*(this->branch))(
            lb[Branching::Front].data(),
            lb[Branching::Back].data(),
            this->IVM->getDepth()
        );
        this->IVM->setDirection(dir);

        //all
        this->IVM->sortSiblingNodes(
            lb[dir],
            prio[dir]
        );
        this->eliminateJobs(lb[dir]);
    };




};

#endif
