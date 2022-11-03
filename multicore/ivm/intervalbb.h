#ifndef INTERVALBB_H_
#define INTERVALBB_H_

// #include "evaluator.h"
#include "branching.h"
#include "pruning.h"
#include "ivm.h"

#include "libbounds.h"

template<typename T>
class Intervalbb{
public:
    Intervalbb(pbab* _pbb);

    bool initAtInterval(std::vector<int>& pos, std::vector<int>& end);
    void setRoot(const int* varOrder, int l1, int l2);

    void run();
    bool next();
    void clear();

    void setBest(const int);
    void eliminateJobs(std::vector<T> lb);
    bool boundLeaf(subproblem& node);

    int first;
    int rootDir;
    std::vector<T> rootRow;

    virtual void boundAndKeepSurvivors(subproblem& subproblem);

    long long int get_leaves_count() const
    {
        return count_leaves;
    }
    long long int get_decomposed_count() const
    {
        return count_decomposed;
    }
    void reset_node_counter(){
        count_leaves = 0;
        count_decomposed = 0;
    }

    std::shared_ptr<ivm> get_ivm(){
        return IVM;
    }
protected:
    pbab* pbb;
    int size;
    std::shared_ptr<ivm> IVM;

    long long int count_leaves;
    long long int count_decomposed;

    std::unique_ptr<Pruning> prune;
    std::unique_ptr<Branching> branch;
    std::unique_ptr<bound_abstract<T>> primary_bound ;

    void unfold();
};

template<typename T>
class IntervalbbIncr : public Intervalbb<T>{
public:
    IntervalbbIncr(pbab* _pbb) : Intervalbb<T>(_pbb)
    {
        secondary_bound = this->pbb->bound_factory->make_bound(_pbb->instance,arguments::secondary_bound);
    };


    void boundAndKeepSurvivors(subproblem& _subpb) override
    {
        std::vector<std::vector<T>> lb(2,std::vector<T>(this->size,0));
        std::vector<std::vector<T>> prio(2,std::vector<T>(this->size,0));

        int dir = this->branch->pre_bound_choice(this->IVM->getDepth());

        if(dir<0){
            //get bounds for both children sets using incremental evaluator
            this->primary_bound->boundChildren(
                    _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                    lb[Branching::Front].data(),lb[Branching::Back].data(),
                    prio[Branching::Front].data(),prio[Branching::Back].data(),this->prune->local_best
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
                    prio[Branching::Front].data(),nullptr,this->prune->local_best
                );
        }else if(dir==Branching::Back){
            this->primary_bound->boundChildren(
                    _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                    nullptr,lb[Branching::Back].data(),
                    nullptr,prio[Branching::Back].data(),this->prune->local_best
                );
        }

        //branching direction was selected
        this->IVM->setDirection(dir);

        //now refine bounds
        //mark all subproblems not pruned by first bound
        std::vector<bool>mask(this->size,false);
        for (int i = _subpb.limit1 + 1; i < _subpb.limit2; i++) {
            int job = _subpb.schedule[i];
            // std::cout<<lb[dir][job]<<" "<<this->prune->local_best<<" "<<(*this->prune)(lb[dir][job])<<"\n";
            if(!(*this->prune)(lb[dir][job])){
                mask[job] = true;
            }
        }
        // std::cout<<"\n";

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


template<typename T>
class IntervalbbEasy : public Intervalbb<T>{
public:
    IntervalbbEasy(pbab* _pbb) : Intervalbb<T>(_pbb){};


    void boundAndKeepSurvivors(subproblem& _subpb) override
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

std::unique_ptr<Intervalbb<int>> make_interval_bb(pbab* pbb, unsigned bound_mode);


//static members
// template<typename T>
// std::vector<T> Intervalbb<T>::rootRow;
// template<typename T>
// int Intervalbb<T>::rootDir = 0;
// template<typename T>
// int Intervalbb<T>::first = true;


#endif
