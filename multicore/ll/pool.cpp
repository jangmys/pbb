#include "pool.h"


Pool::Pool(int _size) : psize(_size)
{
    //data structure used for pool of subproblems (default)
    strategy = DEQUE;
}

//========> cf IVM
void
Pool::setRoot(const int* perm, int l1, int l2)
{
    std::unique_ptr<subproblem> root = std::make_unique<subproblem>(psize);

    for(int i=0;i<psize;i++){
        root->schedule[i]=perm[i];
    }
    root->set_lower_bound(0);
    root->limit1=l1;
    root->limit2=l2;

    push(root);
}








void
Pool::push(std::unique_ptr<subproblem>& p)
{
    switch (strategy) {
        case DEQUE:
            deq.push_front(std::move(p));
            break;
        case STACK:
            pile.push(std::move(p));
            break;
        // case PRIOQ:
        //     pque.push(std::move(p));
        //     break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}
