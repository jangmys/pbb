#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>

//INCLUDE INSTANCES + BOUNDS
#include "../libbounds.h"

template<typename T>
class Foo{
public:
    Foo(){
        std::cout<<"ctor"<<std::endl;
    };
    Foo(bound_abstract<T>& _lb) : bd(){};

// private:
    bound_abstract<T> *bd[2];
};


int main(int argc,char **argv){
    instance_abstract * inst = NULL;
    bound_abstract<int> *bound=NULL;
    bound_abstract<int> *bound2=NULL;
    bound_fsp_weak_idle* bound3=NULL;

    switch(argv[1][0])//DIFFERENT PROBLEMS...
    {
        case 'f': //FLOWSHOP
        {
            switch (argv[2][0]) {//different benchmark sets
                case 't': {
                    inst = new instance_taillard(argv[2]);
                    break;
                }
                case 'V': {
                    inst = new instance_vrf(argv[2]);
                    break;
                }
            }

            //set bound1
            bound_fsp_weak* bd=new bound_fsp_weak();
            bd->init(*inst);
            bound=bd;

            bound_fsp_weak lowerbound;
            lowerbound.init(*inst);

            Foo<int> foo{};
            foo.bd[0] = &lowerbound;
            foo.bd[1] = nullptr;

            //set bound2
            bound_fsp_strong* bd2=new bound_fsp_strong();
            bd2->init(*inst);
            bd2->earlyExit = atoi(argv[3]);
            bd2->machinePairs = atoi(argv[4]);
            bound2=bd2;

            bound_fsp_weak_idle* bd3=new bound_fsp_weak_idle();
            bd3->init(*inst);
            bound3=bd3;

            break;
        }
        // add other problems...
        case 't': //TEST
        {

            break;
        }
    }

    if(!bound)
        return 0;

    int costs[2];

    //a permutation
    std::vector<int> perm(inst->size);
    std::iota(perm.begin(),perm.end(),0);

    //evaluate objective function
    costs[0] = bound->evalSolution(perm.data());
    printf("Makespan LB1: %d\n",costs[0]);

    //evaluate objective function
    costs[0] = bound2->evalSolution(perm.data());
    printf("Makespan LB2: %d\n",costs[0]);



    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm.begin(), perm.end(), g);

    //evaluate 1000000
    std::vector<std::vector<int>>solutions(100000,std::vector<int>(inst->size,0));
    for(auto &p : solutions){
        std::iota(p.begin(),p.end(),0);
    }

    struct timespec t1,t2;

    //empty partial schedules
    int l1=-1;
    int l2=inst->size;

    clock_gettime(CLOCK_MONOTONIC,&t1);
    for(auto &p : solutions){
        bound->bornes_calculer(p.data(), l1+5, l2-5, costs, 99999);
    }
    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9<<"\n";


    clock_gettime(CLOCK_MONOTONIC,&t1);
    for(auto &p : solutions){
        bound2->bornes_calculer(p.data(), l1+5, l2-5, costs, 99999);
    }
    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9<<"\n";


    clock_gettime(CLOCK_MONOTONIC,&t1);
    for(auto &p : solutions){
        bound3->bornes_calculer(p.data(), l1+5, l2-5, costs, 99999);
    }
    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9<<"\n";

    std::cout<<"=======================\n";


    std::vector<int>lb_begin(inst->size,0);
    std::vector<int>lb_end(inst->size,0);
    std::vector<float>prio_begin(inst->size,0);
    std::vector<float>prio_end(inst->size,0);
    std::vector<int>prio_beginI(inst->size,0);
    std::vector<int>prio_endI(inst->size,0);



    clock_gettime(CLOCK_MONOTONIC,&t1);
    for(auto &p : solutions){
        bound->boundChildren(p.data(), l1+5, l2-5, lb_begin.data(), lb_end.data(), prio_beginI.data(), prio_endI.data(),9999999);
    }
    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9<<"\n";



    clock_gettime(CLOCK_MONOTONIC,&t1);
    for(auto &p : solutions){
        bound3->boundChildren(p.data(), l1+5, l2-5, lb_begin.data(), lb_end.data(), prio_begin.data(), prio_end.data());
    }
    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9<<"\n";



    std::iota(perm.begin(),perm.end(),0);
    //LB
    bound->bornes_calculer(perm.data(), l1, l2, costs, 99999);
    printf("LB1 [root]:\t %d\n",costs[0]);

    //LB2
    if(bound2){
        bound2->bornes_calculer(perm.data(), l1, l2, costs, 99999);
        printf("LB2 [root]:\t %d\n",costs[0]);
    }



    l1=0; //fix first job
    l2=inst->size-1; //fix last job
    bound->bornes_calculer(perm.data(), l1, l2, costs, 99999);
    printf("LB1 [+1/-1]:\t %d\n",costs[0]);

    //LB2
    if(bound2){
        bound2->bornes_calculer(perm.data(), l1, l2, costs, 99999);
        printf("LB1 [+1/-1]:\t %d\n",costs[0]);
        // printf("LB2 [root]:\t %d\n",costs[0]);
    }





    l1=-1;
    l2=inst->size;
    for(int i=0;i<inst->size-1;i++){
        l1++;
        bound->bornes_calculer(perm.data(), l1, l2, costs, 99999);
        printf("L1=%d :\t %d\n",l1,costs[0]);
    }

    if(bound2){
        bound2->bornes_calculer(perm.data(), l1, l2, costs, 99999);
        printf("LB2 [+4/-3]:\t %d\n",costs[0]);
    }
    // free(perm);
}
