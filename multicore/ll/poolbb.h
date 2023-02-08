#ifndef POOLBB_H
#define POOLBB_H

#include <memory>
#include <subproblem.h>
#include <pbab.h>
#include <libbounds.h>
#include <pool.h>
#include <mcbb.h>

class Poolbb : public MCbb<int>{
public:
    std::unique_ptr<Pool> pool;

    Poolbb(pbab* _pbb);

    void set_root(subproblem& node);

    virtual std::vector<std::unique_ptr<subproblem>> decompose(subproblem& n);

    void run();
    bool next();

    void set_local_best(int _best){};

    bool isEmpty(){
        return pool->empty();
    };

private:
    int pbsize;
};


class PoolbbEasy : public Poolbb
{
public:
    PoolbbEasy(pbab* _pbb) : Poolbb(_pbb)
    {
        std::cout<<"ctor\n";
    };

    std::vector<std::unique_ptr<subproblem>> decompose(subproblem& n) override
    {
        std::vector<std::unique_ptr<subproblem>>children;
        std::unique_ptr<subproblem> tmp;

        //if only 2 solutions ...
        if (n.simple()) {
            tmp = std::make_unique<subproblem>(n, n.limit1 + 1, Branching::Front);
            tmp->set_lower_bound(primary_bound->evalSolution(tmp->schedule.data()));
            if(!(*prune)(tmp.get()))
                children.push_back(std::move(tmp));

            tmp = std::make_unique<subproblem>(n, n.limit1+2, Branching::Front);
            tmp->set_lower_bound(primary_bound->evalSolution(tmp->schedule.data()));
            if(!(*prune)(tmp.get()))
                children.push_back(std::move(tmp));
        } else {
            std::vector<std::vector<int>> cost(2,std::vector<int>(n.size,0));
            std::vector<std::vector<int>> prio(2,std::vector<int>(n.size,0));

            //a priori choice of branching direction
            auto dir = this->branch->pre_bound_choice(n.depth);

            if(dir<0){
                int tmp_lb[2];
                for (int j = n.limit1 + 1; j < n.limit2; j++) {
                    int job = n.schedule[j];
                    //FRONT
                    std::swap(n.schedule[n.limit1 + 1], n.schedule[j]);
                    this->primary_bound->bornes_calculer(n.schedule.data(), n.limit1 + 1, n.limit2, tmp_lb, this->prune->local_best);
                    cost[0][job]=tmp_lb[0];
                    prio[0][job]=tmp_lb[1];
                    std::swap(n.schedule[n.limit1 + 1], n.schedule[j]);

                    //BACK
                    std::swap(n.schedule[n.limit2 - 1], n.schedule[j]);
                    this->primary_bound->bornes_calculer(n.schedule.data(), n.limit1, n.limit2-1, tmp_lb, this->prune->local_best);
                    cost[1][job]=tmp_lb[0];
                    prio[1][job]=tmp_lb[1];
                    std::swap(n.schedule[n.limit2 - 1], n.schedule[j]);
                }

                dir = (*branch)( cost[0].data(),cost[1].data(),n.depth);

                for (int j = n.limit1 + 1; j < n.limit2; j++) {
                    int job = n.schedule[j];
                    if(!(*prune)(cost[dir][job])){
                        tmp = std::make_unique<subproblem>(n, j, dir);
                        tmp->set_lower_bound(cost[dir][job]);
                        tmp->prio=prio[dir][job];
                        children.push_back(std::move(tmp));
                    }
                }
            }else if(dir==Branching::Front){
                int tmp_lb[2];
                for (int j = n.limit1 + 1; j < n.limit2; j++) {
                    //FRONT
                    std::swap(n.schedule[n.limit1 + 1], n.schedule[j]);
                    this->primary_bound->bornes_calculer(n.schedule.data(), n.limit1 + 1, n.limit2, tmp_lb, this->prune->local_best);

                    if(!(*prune)(tmp_lb[0])){
                        tmp = std::make_unique<subproblem>(n, j, dir);
                        tmp->set_lower_bound(tmp_lb[0]);
                        tmp->prio=tmp_lb[1];
                        children.push_back(std::move(tmp));
                    }
                    std::swap(n.schedule[n.limit1 + 1], n.schedule[j]);
                }
            }else{
                int tmp_lb[2];
                for (int j = n.limit2 - 1; j > n.limit1; j--) {
                    //BACK
                    std::swap(n.schedule[n.limit2 - 1], n.schedule[j]);
                    this->primary_bound->bornes_calculer(n.schedule.data(), n.limit1, n.limit2-1, tmp_lb, this->prune->local_best);

                    if(!(*prune)(tmp_lb[0])){
                        tmp = std::make_unique<subproblem>(n, j, dir);
                        tmp->set_lower_bound(tmp_lb[0]);
                        tmp->prio=tmp_lb[1];
                        children.push_back(std::move(tmp));
                    }
                    std::swap(n.schedule[n.limit2 - 1], n.schedule[j]);
                }
            }
        }
        return children;
    }
};


class PoolbbIncremental : public Poolbb
{
public:
    PoolbbIncremental(pbab* _pbb) : Poolbb(_pbb)
    {};

    std::vector<std::unique_ptr<subproblem>> decompose(subproblem& n) override
    {
        std::vector<std::unique_ptr<subproblem>>children;
        std::unique_ptr<subproblem> tmp;

        //if only 2 solutions ...
        if (n.simple()) {
            tmp = std::make_unique<subproblem>(n, n.limit1 + 1, Branching::Front);
            tmp->set_lower_bound(primary_bound->evalSolution(tmp->schedule.data()));
            if(!(*prune)(tmp.get()))
                children.push_back(std::move(tmp));

            tmp = std::make_unique<subproblem>(n, n.limit1+2, Branching::Front);
            tmp->set_lower_bound(primary_bound->evalSolution(tmp->schedule.data()));
            if(!(*prune)(tmp.get()))
                children.push_back(std::move(tmp));
        } else {
            std::vector<std::vector<int>> cost(2,std::vector<int>(n.size,0));
            std::vector<std::vector<int>> prio(2,std::vector<int>(n.size,0));

            //a priori choice of branching direction
            auto dir = this->branch->pre_bound_choice(n.depth);

            if(dir<0){
                //eval begin-end
                this->primary_bound->boundChildren( n.schedule.data(),n.limit1,n.limit2, cost[0].data(),cost[1].data(),  prio[0].data(),prio[1].data(), this->prune->local_best
                );

                dir = (*branch)( cost[0].data(),cost[1].data(),n.depth);
            }else if(dir == Branching::Front){
                //only begin
                this->primary_bound->boundChildren( n.schedule.data(),n.limit1,n.limit2, cost[0].data(),nullptr,  prio[0].data(),nullptr, this->prune->local_best
                );
            }else{
                //only end
                this->primary_bound->boundChildren( n.schedule.data(),n.limit1,n.limit2, nullptr,cost[1].data(), nullptr,prio[1].data(), this->prune->local_best
                );
            }



            //----------------refine bounds----------------
            //mark all subproblems not pruned by first bound
            std::vector<bool>mask(n.size,false);
            for (int i = n.limit1 + 1; i < n.limit2; i++) {
                int job = n.schedule[i];

                if(!(*this->prune)(cost[dir][job])){
                    mask[job] = true;
                }
            }

            //generate children nodes and refine bounds
            if(dir==Branching::Front){
                int tmp_lb[2];
                for (int j = n.limit1 + 1; j < n.limit2; j++) {
                    int job = n.schedule[j];

                    if(mask[job]){

                        std::swap(n.schedule[n.limit1 + 1], n.schedule[j]);
                        this->secondary_bound->bornes_calculer(n.schedule.data(), n.limit1 + 1, n.limit2, tmp_lb, this->prune->local_best);
                        std::swap(n.schedule[n.limit1 + 1], n.schedule[j]); //swap back

                        if(!(*prune)(tmp_lb[0])){
                            tmp = std::make_unique<subproblem>(n, j, Branching::Front);
                            tmp->set_lower_bound(tmp_lb[0]);
                            tmp->prio=tmp_lb[1];
                            children.push_back(std::move(tmp));
                        }

                    }
                }
            }else{
                int tmp_lb[2];
                for (int j = n.limit2 - 1; j > n.limit1; j--) {
                    int job = n.schedule[j];

                    if(mask[job]){
                        std::swap(n.schedule[n.limit2 - 1], n.schedule[j]);
                        this->secondary_bound->bornes_calculer(n.schedule.data(), n.limit1, n.limit2-1, tmp_lb, this->prune->local_best);
                        std::swap(n.schedule[n.limit2 - 1], n.schedule[j]);

                        if(!(*prune)(tmp_lb[0])){
                            tmp = std::make_unique<subproblem>(n, j, Branching::Back);
                            tmp->set_lower_bound(tmp_lb[0]);
                            tmp->prio=tmp_lb[1];
                            children.push_back(std::move(tmp));
                        }

                    }
                }
            }
        }

        return children;
    }
};

#endif
