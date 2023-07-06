#include "vNNEH.h"
#include "../../common/include/rand.hpp"

void vNNEH::run(std::shared_ptr<subproblem> p, const int N)
{
    m->reset();

    // std::vector<int> jobs(p->schedule);
    std::vector<int> jobs(nbJob);
    std::iota(jobs.begin(),jobs.end(),0);
    util::sort_by_key<int>(jobs,m->sumPT);

    std::vector<int> partial(1,jobs[0]);
    jobs.erase(jobs.begin());

    // std::cout<<jobs.size()<<" "<<partial.size()<<"\n";

    int n_iter = std::rint(std::ceil((float)nbJob/N));

    int mincmax=INT_MAX;

    // for(int i=0;i<n_iter;i++){
    while(!jobs.empty())
    {
        int NN = intRand(1,N);

        size_t take_jobs = std::min((size_t)N,jobs.size());
        std::vector<int> lvn(jobs.begin(),jobs.begin()+take_jobs);
        jobs.erase(jobs.begin(),jobs.begin()+take_jobs);

        while(!lvn.empty())
        {    std::vector<int> jobs(p->schedule);
            mincmax=INT_MAX;
            int inpos=-1;
            int rempos=-1;
            int minjob=-1;

            //evaluate each job in lvn at each position ...
            for(size_t j=0;j<lvn.size();j++){
                // std::cout<<"L_vn["<<lvn[j]<<"] : ";
                std::vector<int> makespans(partial.size()+1,0);

                m->insertMakespans(partial,lvn[j],makespans);

                for(unsigned i=0;i<=partial.size();i++){
                    if(makespans[i]<mincmax){
                        mincmax=makespans[i];
                        inpos=i;
                        minjob=lvn[j];
                        rempos=j;
                    }
                    // std::cout<<makespans[i]<<" ";
                }
                // std::cout<<"\n";
            }

            // std::cout<<"insert job "<<rempos<<"("<<minjob<<") at "<<inpos<<"  = "<<mincmax<<"\n\n";
            m->insert(partial,inpos,minjob);
            lvn.erase(lvn.begin()+rempos);
        }
        std::cout<<take_jobs<<" =========================\n";
    }

    for(size_t i=0;i<partial.size();i++){
        p->schedule[i]=partial[i];
    }
    p->set_fitness(mincmax);
}


void vNNEH::run_me(std::shared_ptr<subproblem> p, const int N)
{
    m->reset();

    std::vector<int> jobs(nbJob);
    std::iota(jobs.begin(),jobs.end(),0);
    util::sort_by_key<int>(jobs,m->sumPT);

    std::vector<int> partial(1,jobs[0]);
    jobs.erase(jobs.begin());

    int mincmax=INT_MAX;

    while(!jobs.empty())
    {
        int NN = intRand(1,N);

        size_t take_jobs = std::min((size_t)N,jobs.size());

        std::vector<int> lvn(jobs.begin(),jobs.begin()+take_jobs);
        jobs.erase(jobs.begin(),jobs.begin()+take_jobs);

        int tmpcmax=0;
        neh->runNEH(lvn,tmpcmax);

        for(auto& j : lvn){
            m->bestInsert(partial, j, mincmax);
        }
    }

    for(size_t i=0;i<partial.size();i++){
        p->schedule[i]=partial[i];
    }
    p->set_fitness(mincmax);
}


void vNNEH::run_plus(std::shared_ptr<subproblem> p, const int N){
    auto best = std::make_shared<subproblem>(nbJob);
    best->set_fitness(INT_MAX);

    auto tmp = std::make_shared<subproblem>(nbJob);
    std::iota(tmp->schedule.begin(),tmp->schedule.end(),0);
    util::sort_by_key<int>(tmp->schedule,m->sumPT);

    for(int i=1;i<N;i+=1){
        run(tmp,i);
        std::cout<<" ? "<<tmp->fitness()<<"\n";
        if(tmp->fitness() < best->fitness()){
            for(size_t i=0;i<tmp->schedule.size();i++){
                best->schedule[i]=tmp->schedule[i];
            }
            best->set_fitness(tmp->fitness());

            std::cout<<"better "<<best->fitness()<<"\n";
        }
    }

    for(size_t i=0;i<best->schedule.size();i++){
        p->schedule[i]=best->schedule[i];
    }
    p->set_fitness(best->fitness());

}