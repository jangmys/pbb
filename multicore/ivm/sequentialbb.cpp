#include <assert.h>

#include "../../common/include/macros.h"

#include "log.h"
#include "pbab.h"
#include "solution.h"
#include "sequentialbb.h"
#include "operator_factory.h"

template<typename T>
sequentialbb<T>::sequentialbb(pbab *_pbb, int _size) :
    pbb(_pbb),size(_size),IVM(std::make_shared<ivm>(size))
{
    count_leaves=0;
    count_decomposed=0;

    pthread_mutex_lock_check(&_pbb->mutex_instance);
    eval = OperatorFactory::createEvaluator(_pbb->instance,0);

    if(rootRow.size()==0)
        rootRow = std::vector<T>(size,0);
    pthread_mutex_unlock(&_pbb->mutex_instance);

    branch = pbb->branching_factory->make_branching(arguments::branchingMode,size,_pbb->initialUB);
    prune = pbb->pruning_factory->make_pruning();

    lower_bound_begin = std::vector<T>(size,0);
    lower_bound_end = std::vector<T>(size,0);

    priority_begin = std::vector<T>(size,0);
    priority_end = std::vector<T>(size,0);
}

template<typename T>
void
sequentialbb<T>::clear()
{
    IVM->clearInterval();
}

template<typename T>
void
sequentialbb<T>::setRoot(const int *varOrder)
{
    clear();

    IVM->setRow(0,varOrder);

    IVM->setDepth(0);
    IVM->decodeIVM();

    pbb->sltn->getBest(prune->local_best);

    IVM->getNode().limit1=-1;
    IVM->getNode().limit2=size;

    if(!first){
        IVM->setRow(0,rootRow.data());
        IVM->setDirection(0,rootDir);
    }else{
        first = false;

        //first line of Matrix
        for(int i=0; i<size; i++){
            IVM->getNode().schedule[i] = pbb->root_sltn->perm[i];
        }
        IVM->setRow(0,pbb->root_sltn->perm);

        //compute children bounds (of IVM->node), choose branching and modify IVM accordingly
        weakBoundPrune();

        //save first line of matrix (bounded root decomposition)
        rootDir = IVM->getDirection(0);
        int c=0;
        for(auto &i : rootRow)
            i=IVM->getCell(0,c++);
        FILE_LOG(logDEBUG) << " === Root Bound: "<<rootDir<<"\n";
    }
}


template<typename T>
void
sequentialbb<T>::initFullInterval()
{
    std::vector<int> zeroFact(size,0);
    std::vector<int> endFact(size,0);

    for (int i = 0; i < size; i++) {
        endFact[i]  = size - i - 1;
    }

    initAtInterval(zeroFact, endFact);
}


template<typename T>
bool
sequentialbb<T>::initAtInterval(std::vector<int> &pos, std::vector<int> &end)
{
    IVM->setDepth(0);

    IVM->setPosition(pos.data());
    IVM->setEnd(end.data());

    if (IVM->beforeEnd()) {
        unfold(arguments::boundMode);
        return true;
    }else{
        return false;
    }
}

template<typename T>
void sequentialbb<T>::setBest(const int bestCost)
{
    prune->local_best = bestCost;
}

template<typename T>
void sequentialbb<T>::run()
{
    while(next());

    pbb->stats.totDecomposed += count_decomposed;
    pbb->stats.leaves += count_leaves;
}



template<typename T>
bool sequentialbb<T>::next()
{
    int state = 0;

    /*this loop decomposes one node, if possible*/
    while (IVM->beforeEnd()) {
        if (IVM->lineEndState()) {
            //backtrack...
            IVM->goUp();
            continue;
        } else if (IVM->pruningCellState()) {
            IVM->goRight();
            continue;
        } else { //if (!IVM->pruningCellState()) {
            state = 1;// exploring
            count_decomposed++;

            IVM->goDown();// branch
            IVM->decodeIVM(); // decode IVM -> subproblems

            if (IVM->isLastLine()) {
                count_leaves++;
                boundLeaf();
                state = 0;
                continue;
            }
            break;
        }
    }

    if(state == 1)
    {
        switch (arguments::boundMode) {
            case 0:
                weakBoundPrune();
                break;
            case 1:
                strongBoundPrune();
                break;
            case 2:
                mixedBoundPrune();
                break;
        }
    }

    return (state == 1);
}

template<typename T>
void
sequentialbb<T>::unfold(int mode)
{
    assert(IVM->intervalValid());

    while (IVM->getDepth() < size - 2) {
        if (IVM->pruningCellState()) {
            IVM->alignLeft();
            break;
        }

        IVM->incrDepth();
        IVM->generateLine(IVM->getDepth(), false);
        IVM->decodeIVM();

        FILE_LOG(logDEBUG) << " === Line: "<<IVM->getDepth()<<"\n";

        switch (mode) {
            case 0:
                weakBoundPrune();
                break;
            case 1:
                strongBoundPrune();
                break;
            case 2:
                mixedBoundPrune();
                break;
        }
    }
} // matrix::unfold



template<typename T>
void
sequentialbb<T>::weakBoundPrune()
{
    // get lower bounds
    eval->get_children_bounds_weak(IVM->getNode(),lower_bound_begin,lower_bound_end,priority_begin,priority_end);
    //make branching decision
    int dir = (*branch)(lower_bound_begin.data(),lower_bound_end.data(),IVM->getDepth());
    IVM->setDirection(IVM->getDepth(),dir);

    if(IVM->getDirection(IVM->getDepth()) == branching::Front){
        IVM->sortSiblingNodes(lower_bound_begin,priority_begin);
        eliminateJobs(lower_bound_begin);
    }else if(IVM->getDirection(IVM->getDepth()) == branching::Back){
        IVM->sortSiblingNodes(lower_bound_end,priority_end);
        eliminateJobs(lower_bound_end);
    }
}

template<typename T>
void
sequentialbb<T>::mixedBoundPrune(){
    // get lower bounds
    eval->get_children_bounds_weak(IVM->getNode(),lower_bound_begin,lower_bound_end,priority_begin,priority_end);
    //make branching decision
    int dir = (*branch)(lower_bound_begin.data(),lower_bound_end.data(),IVM->getDepth());
    IVM->setDirection(IVM->getDepth(),dir);

    boundNode(IVM,lower_bound_begin,lower_bound_end);

    if(IVM->getDirection(IVM->getDepth()) == branching::Front){
        IVM->sortSiblingNodes(lower_bound_begin,priority_begin);
        eliminateJobs(lower_bound_begin);
    }else if(IVM->getDirection(IVM->getDepth()) == branching::Back){
        IVM->sortSiblingNodes(lower_bound_begin,priority_end);
        eliminateJobs(lower_bound_end);
    }
}

template<typename T>
void
sequentialbb<T>::strongBoundPrune(){
    IVM->setDirection(IVM->getDepth(),-1);

    std::fill(lower_bound_begin.begin(),lower_bound_begin.end(),0);
    std::fill(lower_bound_end.begin(),lower_bound_end.end(),0);

    boundNode(IVM,lower_bound_begin,lower_bound_end);
    int dir = (*branch)(lower_bound_begin.data(),lower_bound_end.data(),IVM->getDepth());
    IVM->setDirection(IVM->getDepth(),dir);

    if(IVM->getDirection(IVM->getDepth()) == branching::Front){
        IVM->sortSiblingNodes(lower_bound_begin,priority_begin);
        eliminateJobs(lower_bound_begin);
    }else if(IVM->getDirection(IVM->getDepth()) == branching::Back){
        IVM->sortSiblingNodes(lower_bound_end,priority_end);
        eliminateJobs(lower_bound_end);
    }
}

//compute LB on (left or right) children  of subproblem "node"
//optionally providing :
    //already known bounds
    //current local best (for early stopping of LB calculation)
template<typename T>
void
sequentialbb<T>::computeStrongBounds(subproblem& node, const int be, std::vector<T>& lb){
    std::vector<int>mask(size,0);

    for (int i = node.limit1 + 1; i < node.limit2; i++) {
        int job = node.schedule[i];
        if(!(*prune)(lb[job])){
            mask[job] = 1;
        }
    }

    if(be==branching::Front){
        int fillPos = node.limit1 + 1;
        eval->get_children_bounds_strong(node,mask,be,fillPos,lb,priority_begin,-1);
    }else{
        int fillPos = node.limit2 - 1;
        eval->get_children_bounds_strong(node,mask,be,fillPos,lb,priority_end,-1);
    }
}

template<typename T>
void
sequentialbb<T>::boundNode(std::shared_ptr<ivm> IVM, std::vector<T>& lb_begin, std::vector<T>& lb_end)
{
    int dir=IVM->getDirection(IVM->getDepth());

    FILE_LOG(logDEBUG) << " === Bound: "<<dir<<"\n";

    //
    if (dir == 1){
        computeStrongBounds(IVM->getNode(), branching::Back, lb_end);
    }else if(dir == 0){
        computeStrongBounds(IVM->getNode(), branching::Front, lb_begin);
    }else if(dir == -1){
        // printf("eval BE johnson\n");
        computeStrongBounds(IVM->getNode(), branching::Front, lb_begin);
        computeStrongBounds(IVM->getNode(), branching::Back, lb_end);
    }else{
        perror("boundNode");exit(-1);
    }
}




template<typename T>
bool
sequentialbb<T>::boundLeaf()
{
    bool better=false;
    int cost=eval->getSolutionCost(IVM->getNode());

    if(!(*prune)(cost)){
        better=true;

        //update local best...
        prune->local_best=cost;
        //...and global best (mutex)
        pbb->sltn->update(IVM->getNode().schedule.data(),cost);
        pbb->foundAtLeastOneSolution.store(true);
        pbb->foundNewSolution.store(true);

        //print new best solution
        if(arguments::printSolutions){
            solution tmp(size);
            tmp.update(IVM->getNode().schedule.data(),cost);
            std::cout<<"New Best:\n";
            tmp.print();
        }
    }
    //mark solution as visited
    IVM->eliminateCurrent();

    return better;
}

template<typename T>
void
sequentialbb<T>::eliminateJobs(std::vector<T> lb)
{
    int _line=IVM->getDepth();
    int * jm = IVM->getRowPtr(_line);

    // eliminate
    for (int i = 0; i < size-_line; i++) {
        int job = jm[i];
        if( (*prune)(lb[job]) ){
            jm[i] = negative(job);
        }
    }
}

template class sequentialbb<int>;
