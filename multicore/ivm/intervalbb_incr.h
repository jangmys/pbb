#ifndef INTERVALBB_INCR_H_
#define INTERVALBB_INCR_H_

#include "intervalbb.h"

#include "libbounds.h"

// class pbab;

template<typename T>
class IntervalbbIncr : public Intervalbb<T>{
public:
    IntervalbbIncr(pbab* _pbb) : Intervalbb<T>(_pbb){};


    void boundAndKeepSurvivors(subproblem& _subpb,const int mode) override
    {
        std::vector<std::vector<T>> lb(2,std::vector<T>(this->size,0));
        std::vector<std::vector<T>> prio(2,std::vector<T>(this->size,0));

        int dir = this->branch->pre_bound_choice(this->IVM->getDepth());

        if(dir<0){
            this->eval->get_children_bounds_incr(
                _subpb,
                lb[Branching::Front],lb[Branching::Back],
                prio[Branching::Front],prio[Branching::Back],
                -1
            );

            dir = (*(this->branch))(
                lb[Branching::Front].data(),
                lb[Branching::Back].data(),
                this->IVM->getDepth()
            );
        }else if(dir==Branching::Front){
            this->eval->get_children_bounds_incr(
                _subpb,
                lb[Branching::Front],
                prio[Branching::Front],
                0
            );
        }else if(dir==Branching::Back){
            this->eval->get_children_bounds_incr(
                _subpb,
                lb[Branching::Back],
                prio[Branching::Back],
                1
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


    //weak or mixed bounding
    // if(mode != 2){
    //     // get lower bounds
    //     if(dir == Branching::Front){
    //         eval->get_children_bounds_incr(
    //             _subpb,
    //             lb[Branching::Front],
    //             prio[Branching::Front],
    //             0
    //         );
    //     }
    //     else if(dir == Branching::Back){
    //         eval->get_children_bounds_incr(
    //             _subpb,
    //             lb[Branching::Back],
    //             prio[Branching::Back],
    //             1
    //         );
    //     }
    //     // for full evaluation
    //     // std::vector<bool> mask(size,true);
    //     // eval->get_children_bounds_full(
    //     //     IVM->getNode(),
    //     //     mask, IVM->getNode().limit1 + 1,
    //     //     lb[Branching::Front],
    //     //     prio[Branching::Front],
    //     //     -1, evaluator<T>::Primary);
    // }
    //strong bound only
//     if(mode == 2){
//         std::vector<bool> mask(size,true);
//
//         if(dir == Branching::Front){
//             eval->get_children_bounds_full(
//                 _subpb,
//                 mask, _subpb.limit1 + 1,
//                 lb[Branching::Front],
//                 prio[Branching::Front],
//                 -1, Evaluator<T>::Secondary
//             );
//         }
//         else if(dir == Branching::Back){
//             eval->get_children_bounds_full(
//                 _subpb,
//                 mask, _subpb.limit2 - 1,
//                 lb[Branching::Back],
//                 prio[Branching::Back],
//                 -1, Evaluator<T>::Secondary
//             );
//         }
//     }
//
//     //only mixed
//     if(mode == 1){
//         dir = IVM->getDirection();
//         refineBounds(
//             _subpb,
//             dir,
//             lb[dir],
//             prio[dir]
//         );
//     }
//
//     //all
//     dir = IVM->getDirection();
//     IVM->sortSiblingNodes(
//         lb[dir],
//         prio[dir]
//     );
//     eliminateJobs(lb[dir]);
//
};

#endif
