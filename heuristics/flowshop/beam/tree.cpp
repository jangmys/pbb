#include "macros.h"
#include "subproblem.h"
#include "log.h"

#include "tree.h"

#include "libbounds.h"

#include "operator_factory.h"


Tree::Tree(instance_abstract* inst, int _size) : psize(_size)
{
    //data structure used for pool of subproblems (default)
    strategy = DEQUE;
}

// Tree::~Tree()
// {
//
// }

void
Tree::setRoot(std::vector<int> perm, int l1, int l2)
{
    std::shared_ptr<subproblem> root = std::make_shared<subproblem>(psize);

    for(int i=0;i<psize;i++){
        root->schedule[i]=perm[i];
    }
    root->set_lower_bound(0);
    root->limit1=l1;
    root->limit2=l2;

    push(root);
}

//=============================================================================================
//================================================= gestion pools ===========================
//=============================================================================================
//======================================================================
int
Tree::size()
{
    switch (strategy) {
        case DEQUE:
            return deq.size();
        case STACK:
            return pile.size();
        case PRIOQ:
            return pque.size();
        default:
            std::cout << "Undefined DS";
            exit(1);
    }
}

void
Tree::push_back(std::shared_ptr<subproblem> p)
{
    deq.push_back(p);
}

void
Tree::push(std::shared_ptr<subproblem> p)
{
    switch (strategy) {
        case DEQUE:
            deq.push_front(p);
            break;
        case STACK:
            pile.push(p);
            break;
        case PRIOQ:
            pque.push(p);
            break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

void
Tree::pop()
{
    switch (strategy) {
        case DEQUE:
            deq.pop_front();
            break;
        case STACK:
            pile.pop();
            break;
        case PRIOQ:
            pque.pop();
            break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

std::shared_ptr<subproblem>
Tree::top()
{
    switch (strategy) {
        case DEQUE:
            return deq.front();
        case STACK:
            return pile.top();
        case PRIOQ:
            return pque.top();
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

bool
Tree::empty()
{
    switch (strategy){
        case DEQUE:
            return deq.empty();
        case STACK:
            return pile.empty();
        case PRIOQ:
            return pque.empty();
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

std::shared_ptr<subproblem>
Tree::take()//take problem from top
{
    std::shared_ptr<subproblem> n=(empty()) ? NULL : top();
    if(n) pop();

    return n;
}

void
Tree::insert(std::vector<std::shared_ptr<subproblem>>&ns)
{
    //no children (decomposition avoid generation of unpromising children)
    if (!ns.size())
        return;

    //children inserted with push_back [ 1 2 3 ... ]
    //for left->right exploration, insert (push) in reverse order
    for (auto i = ns.rbegin(); i != ns.rend(); i++) {
        push(std::move(*i));
    }
}


void
Tree::clearPool()
{
    while(!empty()){
        pop();
    }
}
