#include <assert.h>

#include "../../common/include/macros.h"

#include "log.h"
#include "pbab.h"
#include "solution.h"
#include "intervalbb.h"
#include "operator_factory.h"

template<typename T>
Intervalbb<T>::Intervalbb(pbab *_pbb) :
    pbb(_pbb),size(_pbb->size),IVM(std::make_shared<ivm>(size)),count_leaves(0),count_decomposed(0)
{
    //why not pass operators to the ctor?
    prune = pbb->pruning_factory->make_pruning();
    branch = pbb->branching_factory->make_branching();

    pthread_mutex_lock_check(&_pbb->mutex_instance);
    eval = OperatorFactory::createEvaluator(
        pbb->bound_factory->make_bound(_pbb->instance,0),
        pbb->bound_factory->make_bound(_pbb->instance,1));
    if(rootRow.size()==0)
        rootRow = std::vector<T>(size,0);
    pthread_mutex_unlock(&_pbb->mutex_instance);
}

template<typename T>
void
Intervalbb<T>::clear()
{
    IVM->clearInterval();
}

template<typename T>
void
Intervalbb<T>::setRoot(const int *varOrder,int l1,int l2)
{
    clear();

    pbb->sltn->getBest(prune->local_best);

    IVM->setRow(0,varOrder);

    IVM->setDepth(0);
    IVM->decodeIVM();
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

        //compute children bounds (of IVM->node), choose Branching and modify IVM accordingly
        boundAndKeepSurvivors(IVM->getNode(),arguments::boundMode);

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
Intervalbb<T>::initFullInterval()
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
Intervalbb<T>::initAtInterval(std::vector<int> &pos, std::vector<int> &end)
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
void Intervalbb<T>::setBest(const int bestCost)
{
    prune->local_best = bestCost;
}

template<typename T>
void Intervalbb<T>::run()
{
    while(next());

    pbb->stats.totDecomposed += count_decomposed;
    pbb->stats.leaves += count_leaves;
}



template<typename T>
void Intervalbb<T>::boundAndKeepSurvivors_static(subproblem& _subpb, const int mode)
{
    std::vector<std::vector<T>> lb(2,std::vector<T>(size,0));
    std::vector<std::vector<T>> prio(2,std::vector<T>(size,0));

    auto dir = (*branch)(
        nullptr,nullptr,
        IVM->getDepth()
    );
    IVM->setDirection(dir);

    //weak or mixed bounding
    if(mode != 2){
        // get lower bounds
        if(dir == Branching::Front){
            eval->get_children_bounds_incr(
                _subpb,
                lb[Branching::Front],
                prio[Branching::Front],
                0
            );
        }
        else if(dir == Branching::Back){
            eval->get_children_bounds_incr(
                _subpb,
                lb[Branching::Back],
                prio[Branching::Back],
                1
            );
        }
        // for full evaluation
        // std::vector<bool> mask(size,true);
        // eval->get_children_bounds_full(
        //     IVM->getNode(),
        //     mask, IVM->getNode().limit1 + 1,
        //     lb[Branching::Front],
        //     prio[Branching::Front],
        //     -1, evaluator<T>::Primary);
    }
    //strong bound only
    if(mode == 2){
        std::vector<bool> mask(size,true);

        int best = prune->local_best;

        if(dir == Branching::Front){
            eval->get_children_bounds_full(
                _subpb,
                mask, _subpb.limit1 + 1,
                lb[Branching::Front],
                prio[Branching::Front],
                best, Evaluator<T>::Secondary
            );
        }
        else if(dir == Branching::Back){
            eval->get_children_bounds_full(
                _subpb,
                mask, _subpb.limit2 - 1,
                lb[Branching::Back],
                prio[Branching::Back],
                best, Evaluator<T>::Secondary
            );
        }
    }

    //only mixed
    if(mode == 1){
        dir = IVM->getDirection();
        refineBounds(
            _subpb,
            dir,
            lb[dir],
            prio[dir]
        );
    }

    //all
    dir = IVM->getDirection();
    IVM->sortSiblingNodes(
        lb[dir],
        prio[dir]
    );
    eliminateJobs(lb[dir]);
}


template<typename T>
void Intervalbb<T>::boundAndKeepSurvivors(subproblem& _subpb, const int mode)
{
    std::vector<std::vector<T>> lb(2,std::vector<T>(size,0));
    std::vector<std::vector<T>> prio(2,std::vector<T>(size,0));

    //weak or mixed bounding
    if(mode != 2){
        // get lower bounds
        eval->get_children_bounds_incr(
            _subpb,
            lb[Branching::Front],
            lb[Branching::Back],
            prio[Branching::Front],
            prio[Branching::Back],
            branch->get_type()
        );

        // for full evaluation
        // std::vector<bool> mask(size,true);
        // eval->get_children_bounds_full(
        //     IVM->getNode(),
        //     mask, IVM->getNode().limit1 + 1,
        //     lb[Branching::Front],
        //     prio[Branching::Front],
        //     -1, evaluator<T>::Primary);
    }
    //strong bound only
    if(mode == 2){
        std::vector<bool> mask(size,true);

        eval->get_children_bounds_full(
            _subpb,
            mask, _subpb.limit1 + 1,
            lb[Branching::Front],
            prio[Branching::Front],
            -1, Evaluator<T>::Secondary);
        eval->get_children_bounds_full(
            _subpb,
            mask, _subpb.limit2 - 1,
            lb[Branching::Back],
            prio[Branching::Back],
            -1, Evaluator<T>::Secondary);
    }

    //all
    {
        //make Branching decision
        auto dir = (*branch)(
            lb[Branching::Front].data(),
            lb[Branching::Back].data(),
            IVM->getDepth()
        );
        IVM->setDirection(dir);
    }

    //only mixed
    if(mode == 1){
        auto dir = IVM->getDirection();
        refineBounds(
            _subpb,
            dir,
            lb[dir],
            prio[dir]
        );
    }

    //all
    auto dir = IVM->getDirection();
    IVM->sortSiblingNodes(
        lb[dir],
        prio[dir]
    );
    eliminateJobs(lb[dir]);
}



template<typename T>
bool Intervalbb<T>::next()
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
                boundLeaf(IVM->getNode());
                state = 0;
                continue;
            }
            break;
        }
    }

    //bound, set Branching direction, prune
    if(state == 1)
    {
        if(arguments::branchingMode < 0){
            boundAndKeepSurvivors_static(IVM->getNode(),arguments::boundMode);
        }else{
            boundAndKeepSurvivors(IVM->getNode(),arguments::boundMode);
        }
    }

    return (state == 1);
}

template<typename T>
void
Intervalbb<T>::unfold(int mode)
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

        FILE_LOG(logDEBUG) << " === Unfold line: "<<IVM->getDepth()<<"\n";

        boundAndKeepSurvivors(IVM->getNode(),mode);
    }
} // matrix::unfold

/**
 * compute LB on (left or right) children of subproblem "node" using single-child bounder
 *
 * @param node parent subproblem
 * @param be =0 iff front, = 1 iff back
 * @param lb (inout) already known bounds (e.g. precomputed by weaker bound, 0 if not). will be overwritten
 * @param prio (out) children priority values
 */
template<typename T>
void
Intervalbb<T>::refineBounds(subproblem& node, const int be,
    std::vector<T>& lb,
    std::vector<T>& prio){
    std::vector<bool>mask(size,false);

    for (int i = node.limit1 + 1; i < node.limit2; i++) {
        int job = node.schedule[i];
        if(!(*prune)(lb[job])){
            mask[job] = true;
        }
    }

    if(be==Branching::Front){
        eval->get_children_bounds_full(
            node,mask,node.limit1 + 1,lb,prio,-1,Evaluator<T>::Secondary
        );
    }else{
        eval->get_children_bounds_full(
            node,mask,node.limit2 - 1,lb,prio,-1,Evaluator<T>::Secondary
        );
    }
}


template<typename T>
bool
Intervalbb<T>::boundLeaf(subproblem& node)
{
    bool better=false;
    int cost=eval->get_solution_cost(node);

    if(!(*prune)(cost)){
        better=true;

        //update local best...
        prune->local_best=cost;
        //...and global best (mutex)
        pbb->sltn->update(node.schedule.data(),cost);
        pbb->foundAtLeastOneSolution.store(true);
        pbb->foundNewSolution.store(true);

        //print new best solution
        if(arguments::printSolutions){
            solution tmp(size);
            tmp.update(node.schedule.data(),cost);
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
Intervalbb<T>::eliminateJobs(std::vector<T> lb)
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

template class Intervalbb<int>;
