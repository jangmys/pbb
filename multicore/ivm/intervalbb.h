/*
Interval (IVM)-based BB
*/
#ifndef INTERVALBB_H_
#define INTERVALBB_H_

// #include "evaluator.h"
#include "branching.h"
#include "pruning.h"
#include "ivm.h"

#include "mcbb.h"

#include "libbounds.h"

//base class for interval-based CPU-BB
//inherits from MCbb (DS-agnostic)
//child classes override boundAndKeepSurvivors (decomposition method)
template<typename T>
class Intervalbb : public MCbb<T>{
public:
    Intervalbb(pbab* _pbb);

    bool initAtInterval(std::vector<int>& pos, std::vector<int>& end);

    void setRoot(const int* varOrder);
    void setRoot(const std::vector<int> varOrder);


    void run();
    bool next();
    void clear();

    void setLocalBest(const int best){
        setBest(best);
    }
    void setBest(const int);

    void eliminateJobs(std::vector<T> lb);
    bool boundLeaf(subproblem& node);

    int first;
    int rootDir;
    std::vector<T> rootRow;

    virtual void boundAndKeepSurvivors(subproblem& subproblem);

    std::shared_ptr<IVM> get_ivm(){
        return _IVM;
    }

    subproblem&
    getNode()
    {
        return get_ivm()->getNode();
    }

    void
    getInterval(int *pos,int *end){
        get_ivm()->getInterval(pos, end);
    }

    bool print_new_solutions = false;
protected:
    pbab* pbb;
    int size;
    std::shared_ptr<IVM> _IVM;

    void unfold();
};

template<typename T>
class IntervalbbIncr : public Intervalbb<T>{
public:
    IntervalbbIncr(pbab* _pbb) : Intervalbb<T>(_pbb)
    {};

    void boundAndKeepSurvivors(subproblem& _subpb) override
    {
        std::vector<std::vector<T>> lb(2,std::vector<T>(this->size,0));
        std::vector<std::vector<T>> prio(2,std::vector<T>(this->size,0));

        int dir = this->branch->pre_bound_choice(this->_IVM->getDepth());

        if(dir<0){
            //get bounds for both children sets using incremental evaluator
            this->primary_bound->boundChildren(
                    _subpb.schedule,_subpb.limit1,_subpb.limit2,
                    lb[Branching::Front].data(),lb[Branching::Back].data(),
                    prio[Branching::Front].data(),prio[Branching::Back].data(),this->prune->local_best
                );

            //choose branching direction
            dir = (*(this->branch))(
                lb[Branching::Front].data(),
                lb[Branching::Back].data(),
                this->_IVM->getDepth()
            );
        }else if(dir==Branching::Front){
            this->primary_bound->boundChildren(
                    _subpb.schedule,_subpb.limit1,_subpb.limit2,
                    lb[Branching::Front].data(),nullptr,
                    prio[Branching::Front].data(),nullptr,this->prune->local_best
                );
        }else if(dir==Branching::Back){
            this->primary_bound->boundChildren(
                    _subpb.schedule,_subpb.limit1,_subpb.limit2,
                    nullptr,lb[Branching::Back].data(),
                    nullptr,prio[Branching::Back].data(),this->prune->local_best
                );
        }

        //branching direction was selected
        this->_IVM->setDirection(dir);

        // std::cout<<"refine bounds\n";

        //now refine bounds
        //mark all subproblems not pruned by first bound
        std::vector<bool>mask(this->size,false);
        for (int i = _subpb.limit1 + 1; i < _subpb.limit2; i++) {
            int job = _subpb.schedule[i];
            // std::cout<<job<<" "<<lb[dir][job]<<" "<<this->prune->local_best<<" "<<(*this->prune)(lb[dir][job])<<"\n";
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
                    this->secondary_bound->bornes_calculer(_subpb.schedule, _subpb.limit1 + 1, _subpb.limit2, costs, this->prune->local_best);
                    lb[Branching::Front][job] = costs[0];
                    prio[Branching::Front][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                }
                // std::cout<<job<<"\t\t"<<lb[dir][job]<<" "<<this->prune->local_best<<" "<<(*this->prune)(lb[dir][job])<<"\n";
            }
        }else{
            int costs[2];
            for (int i = _subpb.limit2 - 1; i > _subpb.limit1; i--) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                    this->secondary_bound->bornes_calculer(_subpb.schedule, _subpb.limit1, _subpb.limit2-1, costs, this->prune->local_best);
                    lb[Branching::Back][job] = costs[0];
                    prio[Branching::Back][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                }
                // std::cout<<job<<"\t\t"<<lb[dir][job]<<" "<<this->prune->local_best<<" "<<(*this->prune)(lb[dir][job])<<"\n";
            }
        }

        this->_IVM->sortSiblingNodes(
            lb[dir],
            prio[dir]
        );
        this->eliminateJobs(lb[dir]);
    }
};


template<typename T>
class IntervalbbEasy : public Intervalbb<T>{
public:
    IntervalbbEasy(pbab* _pbb) : Intervalbb<T>(_pbb){};


    void boundAndKeepSurvivors(subproblem& _subpb) override
    {
        std::vector<std::vector<T>> lb(2,std::vector<T>(this->size,0));
        std::vector<std::vector<T>> prio(2,std::vector<T>(this->size,0));

        //a priori choice of branching direction
        int dir = this->branch->pre_bound_choice(this->_IVM->getDepth());
        std::vector<bool> mask(this->size,true);

        if(dir<0){
            //evaluate both children sets using full LB evalution function
            int costs[2];

            for (int i = _subpb.limit1 + 1; i < _subpb.limit2; i++) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    //front
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                    this->primary_bound->bornes_calculer(_subpb.schedule, _subpb.limit1 + 1, _subpb.limit2, costs, this->prune->local_best);
                    lb[Branching::Front][job] = costs[0];
                    prio[Branching::Front][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                    //back
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                    this->primary_bound->bornes_calculer(_subpb.schedule, _subpb.limit1, _subpb.limit2 - 1, costs, this->prune->local_best);
                    lb[Branching::Back][job] = costs[0];
                    prio[Branching::Back][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                }
            }

            //make Branching decision
            dir = (*this->branch)(
                lb[Branching::Front].data(),
                lb[Branching::Back].data(),
                this->_IVM->getDepth()
            );
            // this->_IVM->setDirection(dir);
        }else if(dir==Branching::Front){
            int costs[2];
            for (int i = _subpb.limit1 + 1; i < _subpb.limit2; i++) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                    this->primary_bound->bornes_calculer(_subpb.schedule, _subpb.limit1 + 1, _subpb.limit2, costs, this->prune->local_best);
                    lb[Branching::Front][job] = costs[0];
                    prio[Branching::Front][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit1 + 1], _subpb.schedule[i]);
                }
            }
            // this->_IVM->setDirection(dir);
        }else{
            int costs[2];
            for (int i = _subpb.limit2 - 1; i > _subpb.limit1; i--) {
                int job = _subpb.schedule[i];
                if(mask[job]){
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                    this->primary_bound->bornes_calculer(_subpb.schedule, _subpb.limit1, _subpb.limit2-1, costs, this->prune->local_best);
                    lb[Branching::Back][job] = costs[0];
                    prio[Branching::Back][job]=costs[1];
                    std::swap(_subpb.schedule[_subpb.limit2 - 1], _subpb.schedule[i]);
                }
            }
        }
        this->_IVM->setDirection(dir);

        this->_IVM->sortSiblingNodes(
            lb[dir],
            prio[dir]
        );
        this->eliminateJobs(lb[dir]);
    }
};


#endif
