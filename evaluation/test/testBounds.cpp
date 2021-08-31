#include <iostream>

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
            bd->init(inst);
            // bd->branchingMode = atoi(argv[3]);
            bound=bd;

            bound_fsp_weak lowerbound;
            lowerbound.init(inst);

            Foo<int> foo{};
            foo.bd[0] = &lowerbound;
            foo.bd[1] = nullptr;

            //set bound2
            bound_fsp_strong* bd2=new bound_fsp_strong();
            bd2->init(inst);
            // bd2->branchingMode = atoi(argv[3]);
            bd2->earlyExit = atoi(argv[3]);
            bd2->machinePairs = atoi(argv[4]);
            bound2=bd2;

            break;
        }
        // add other problems...
        case 't': //TEST
        {

            break;
        }
    }

    int costs[2];

    //a permutation
    std::vector<int> perm(inst->size);

    for(int i=0;i<inst->size;i++){
        perm[i]=i;
    }

    if(!bound)
        return 0;

    //evaluate objective function
    costs[0] = bound->evalSolution(perm.data());
    printf("Makespan: %d\n",costs[0]);

    //empty partial schedules
    int l1=-1;
    int l2=inst->size;

    //LB
    bound->bornes_calculer(perm.data(), l1, l2, costs, 99999);
    printf("LB1 [root]:\t %d\n",costs[0]);

    //LB2
    if(bound2){
        bound2->bornes_calculer(perm.data(), l1, l2, costs, 99999);
        printf("LB2 [root]:\t %d\n",costs[0]);
    }

    l1=3; //fix first job
    l2=inst->size-3; //fix last job
    bound->bornes_calculer(perm.data(), l1, l2, costs, 99999);
    printf("LB1 [+4/-3]:\t %d\n",costs[0]);

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
