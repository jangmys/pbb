#include "pool.h"


Pool::Pool(int _size) : psize(_size)
{
    //data structure used for pool of subproblems (default)
    strategy = DEQUE;

    std::cout<<"Pool ctor "<<psize<<"\n";
}

//========> cf IVM
void
Pool::set_root(const int* perm, int l1, int l2)
{
    std::unique_ptr<subproblem> root = std::make_unique<subproblem>(psize);

    for(int i=0;i<psize;i++){
        root->schedule[i]=perm[i];
    }
    root->set_lower_bound(0);
    root->limit1=l1;
    root->limit2=l2;

    std::cout<<"push root : "<<*root<<"\n";
    push(std::move(root));
    std::cout<<"pushed root "<<size()<<"\n";
}

void
Pool::push(std::unique_ptr<subproblem> p)
{
    // std::cout<<"push\n";

    switch (strategy) {
        case DEQUE:
            deq.push_front(std::move(p));
            break;
        case STACK:
            pile.push(std::move(p));
            break;
        // case PRIOQ:
        //     pque.push(p);
        //     break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

std::unique_ptr<subproblem>
Pool::take()
{
    std::unique_ptr<subproblem> n=(empty()) ? NULL : std::move(top());
    if(n) pop();

    return std::move(n);
}

std::unique_ptr<subproblem>
Pool::top()
{
    switch (strategy) {
        case DEQUE:
            return std::move(deq.front());
        case STACK:
            return std::move(pile.top());
        // case PRIOQ:
        //     return pque.top();
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

void
Pool::pop()
{
    switch (strategy) {
        case DEQUE:
            deq.pop_front();
            break;
        case STACK:
            pile.pop();
            break;
        // case PRIOQ:
        //     pque.pop();
        //     break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

bool
Pool::empty()
{
    switch (strategy){
        case DEQUE:
            return deq.empty();
        case STACK:
            return pile.empty();
        // case PRIOQ:
        //     return pque.empty();
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

int
Pool::size()
{
    switch (strategy) {
        case DEQUE:
            return deq.size();
        case STACK:
            return pile.size();
        // case PRIOQ:
        //     return pque.size();
        default:
            std::cout << "Undefined DS";
            exit(1);
    }
}
