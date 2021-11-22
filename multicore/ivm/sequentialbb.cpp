#include "../../common/include/macros.h"

#include "log.h"
#include "pbab.h"
#include "solution.h"
#include "sequentialbb.h"
#include "operator_factory.h"

template<typename T>
sequentialbb<T>::sequentialbb(pbab *_pbb, int _size)
{
    pbb = _pbb;
    size=_size;

    IVM = new ivm(size);

    count_leaves=0;
    count_decomposed=0;

    pthread_mutex_lock_check(&_pbb->mutex_instance);
    // bound.push_back(std::move(OperatorFactory::createBound(_pbb->instance,0)));
    // bound.push_back(std::move(OperatorFactory::createBound(_pbb->instance,1)));

    eval = OperatorFactory::createEvaluator(_pbb->instance,0);

    if(rootRow.size()==0)
        rootRow = std::vector<T>(size,0);
    pthread_mutex_unlock(&_pbb->mutex_instance);

    branch = OperatorFactory::createBranching(arguments::branchingMode,size,_pbb->initialUB);
    prune = OperatorFactory::createPruning(arguments::findAll);

    lower_bound_begin = std::vector<T>(size,0);
    lower_bound_end = std::vector<T>(size,0);

    priority_begin = std::vector<T>(size,0);
    priority_end = std::vector<T>(size,0);

}

template<typename T>
sequentialbb<T>::~sequentialbb()
{
    delete IVM;
    // delete bd;
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

    for(int i=0; i<size; i++){
        IVM->jobMat[i] = varOrder[i];
    }
    IVM->line=0;

    IVM->decodeIVM();
    boundRoot();
    // bd->boundRoot(IVM);
}

template<typename T>
bool
sequentialbb<T>::solvedAtRoot()
{
    bool solved=true;
    for(int i=0;i<size;i++){
        solved &= (IVM->jobMat[i]<0);
    }
    if(solved){
        printf("problem solved at level 0\n");
        for(int i=0; i<size; i++){
            std::cout<<IVM->jobMat[i]<<" ";
        }
        std::cout<<std::endl;
        // IVM->posVect[0]=size;
    }
    return solved;
}

template<typename T>
void
sequentialbb<T>::initFullInterval()
{
    int * zeroFact = (int *) malloc(size * sizeof(int));
    int * endFact  = (int *) malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        zeroFact[i] = 0;
        endFact[i]  = size - i - 1;
    }

    initAtInterval(zeroFact, endFact);

    free(zeroFact);
    free(endFact);
}


template<typename T>
bool
sequentialbb<T>::initAtInterval(int * pos, int * end)
{
    int l = 0;
    IVM->line = l;

    for (int i = 0; i < size; i++) {
        IVM->posVect[i] = pos[i];
        IVM->endVect[i] = end[i];
    }

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
        } else if (!IVM->pruningCellState()) {
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

        if (IVM->line >= size - 1) {
            printf("too deeep\n");
            exit(0);
        }
    }

    return (state == 1);
}

template<typename T>
void
sequentialbb<T>::unfold(int mode)
{
    if(!IVM->intervalValid()){
        std::cout<<"interval invalid\n";
        exit(-1);
    }

    while (IVM->line < size - 2) {
        if (IVM->pruningCellState()) {
            for (int i = IVM->line + 1; i < size; i++) {
                IVM->posVect[i] = 0;
            }
            break;
        }

        IVM->line++;
        IVM->generateLine(IVM->line, false);
        IVM->decodeIVM();

        FILE_LOG(logDEBUG) << " === Line: "<<IVM->line<<"\n";

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
    eval->get_children_bounds_weak(*(IVM->node.get()),lower_bound_begin,lower_bound_end,priority_begin,priority_end);
    //make branching decision
    IVM->dirVect[IVM->line] = (*branch)(lower_bound_begin.data(),lower_bound_end.data(),IVM->line);

    if(IVM->dirVect[IVM->line] == branching::Front){
        IVM->sortSiblingNodes(lower_bound_begin,priority_begin);
        eliminateJobs(lower_bound_begin);
    }else if(IVM->dirVect[IVM->line] == branching::Back){
        IVM->sortSiblingNodes(lower_bound_end,priority_end);
        eliminateJobs(lower_bound_end);
    }
}

template<typename T>
void
sequentialbb<T>::mixedBoundPrune(){
    // get lower bounds
    eval->get_children_bounds_weak(*(IVM->node.get()),lower_bound_begin,lower_bound_end,priority_begin,priority_end);
    //make branching decision
    IVM->dirVect[IVM->line]=(*branch)(lower_bound_begin.data(),lower_bound_end.data(),IVM->line);

    boundNode(IVM,lower_bound_begin,lower_bound_end);

    if(IVM->dirVect[IVM->line] == branching::Front){
        IVM->sortSiblingNodes(lower_bound_begin,priority_begin);
        eliminateJobs(lower_bound_begin);
    }else if(IVM->dirVect[IVM->line] == branching::Back){
        IVM->sortSiblingNodes(lower_bound_begin,priority_end);
        eliminateJobs(lower_bound_end);
    }
}

template<typename T>
void
sequentialbb<T>::strongBoundPrune(){
    IVM->dirVect[IVM->line]=-1;
    std::fill(lower_bound_begin.begin(),lower_bound_begin.end(),0);
    std::fill(lower_bound_end.begin(),lower_bound_end.end(),0);

    boundNode(IVM,lower_bound_begin,lower_bound_end);
    IVM->dirVect[IVM->line]=(*branch)(lower_bound_begin.data(),lower_bound_end.data(),IVM->line);

    if(IVM->dirVect[IVM->line] == branching::Front){
        IVM->sortSiblingNodes(lower_bound_begin,priority_begin);
        eliminateJobs(lower_bound_begin);
    }else if(IVM->dirVect[IVM->line] == branching::Back){
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
sequentialbb<T>::computeStrongBounds(subproblem* node, const int be, std::vector<T>& lb){
    std::vector<int>mask(size,0);

    for (int i = node->limit1 + 1; i < node->limit2; i++) {
        int job = node->schedule[i];
        if(!(*prune)(lb[job])){
            mask[job] = 1;
        }
    }

    if(be==branching::Front){
        int fillPos = node->limit1 + 1;
        eval->get_children_bounds_strong(*node,mask,be,fillPos,lb,priority_begin,-1);
    }else{
        int fillPos = node->limit2 - 1;
        eval->get_children_bounds_strong(*node,mask,be,fillPos,lb,priority_end,-1);
    }
}

template<typename T>
void
sequentialbb<T>::boundNode(const ivm* IVM, std::vector<T>& lb_begin, std::vector<T>& lb_end)
{
    int dir=IVM->dirVect[IVM->line];

    FILE_LOG(logDEBUG) << " === Bound: "<<dir<<"\n";

    //
    if (dir == 1){
        computeStrongBounds(IVM->node.get(), branching::Back, lb_end);
    }else if(dir == 0){
        computeStrongBounds(IVM->node.get(), branching::Front, lb_begin);
    }else if(dir == -1){
        // printf("eval BE johnson\n");
        computeStrongBounds(IVM->node.get(), branching::Front, lb_begin);
        computeStrongBounds(IVM->node.get(), branching::Back, lb_end);
    }else{
        perror("boundNode");exit(-1);
    }
}




template<typename T>
bool
sequentialbb<T>::boundLeaf()
{
    bool better=false;
    int cost;

    cost=eval->getSolutionCost(*IVM->node.get());

    if(!(*prune)(cost)){
        better=true;

        //update local best...
        prune->local_best=cost;
        //...and global best (mutex)
        pbb->sltn->update(IVM->node->schedule.data(),cost);
        pbb->foundAtLeastOneSolution.store(true);
        pbb->foundNewSolution.store(true);

        //print new best solution
        if(arguments::printSolutions){
            solution *tmp = new solution(size);
            tmp->update(IVM->node->schedule.data(),cost);
            std::cout<<"New Best:\n";
            tmp->print();
            delete tmp;
        }
    }
    //mark solution as visited
    int pos = IVM->posVect[IVM->line];
    int job = IVM->jobMat[IVM->line * size + pos];
    IVM->jobMat[IVM->line * size + pos] = negative(job);

    return better;
}

template<typename T>
void
sequentialbb<T>::boundRoot(){
	pbb->sltn->getBest(prune->local_best);

    IVM->node->limit1=-1;
    IVM->node->limit2=size;

    if(!first){
        int c=0;
        for(auto i : rootRow)
            IVM->jobMat[c++]=i;
        IVM->dirVect[0] = rootDir;
    }else{
        first = false;

        //first line of Matrix
        for(int i=0; i<size; i++){
            IVM->node->schedule[i] = pbb->root_sltn->perm[i];
            IVM->jobMat[i] = pbb->root_sltn->perm[i];
        }
        IVM->line=0;
        IVM->node->limit1=-1;
        IVM->node->limit2=size;

        weakBoundPrune();

        //save first line of matrix (bounded root decomposition)
        rootDir = IVM->dirVect[0];
        int c=0;
        for(auto &i : rootRow)
            i=IVM->jobMat[c++];

        FILE_LOG(logDEBUG) << " === Root Bound: "<<rootDir<<"\n";
    }
}


template<typename T>
void
sequentialbb<T>::eliminateJobs(std::vector<T> lb)
{
    int _line=IVM->line;
    int * jm = IVM->jobMat + _line * size;

    // eliminate
    for (int i = 0; i < size-_line; i++) {
        int job = jm[i];
        if( (*prune)(lb[job]) ){
            jm[i] = negative(job);
        }
    }
}

template class sequentialbb<int>;
