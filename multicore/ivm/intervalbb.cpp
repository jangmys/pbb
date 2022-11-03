#include <assert.h>

#include "../../common/include/macros.h"

#include "log.h"
#include "pbab.h"
#include "solution.h"
#include "intervalbb.h"
#include "operator_factory.h"

std::unique_ptr<Intervalbb<int>> make_interval_bb(pbab* pbb, unsigned bound_mode)
{
    if(bound_mode == 0){
        return std::make_unique<Intervalbb<int>>(pbb);
    }else if(bound_mode == 1){
        return std::make_unique<IntervalbbEasy<int>>(pbb);
    }else{
        return std::make_unique<IntervalbbIncr<int>>(pbb);
    }
}

template<typename T>
Intervalbb<T>::Intervalbb(pbab *_pbb) : first(true), pbb(_pbb), size(_pbb->size),IVM(std::make_shared<ivm>(size)),count_leaves(0),count_decomposed(0)
{
    //why not pass operators to the ctor?
    prune = pbb->pruning_factory->make_pruning();
    branch = pbb->branching_factory->make_branching();
    primary_bound = pbb->bound_factory->make_bound(pbb->instance,arguments::primary_bound);

    rootRow = std::vector<T>(size,0);
    pthread_mutex_init(&first_mutex,NULL);
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
    // IVM->clearInterval();
    IVM->setDepth(0);

    if(!first){
        //row 0 and direction have been saved
        IVM->setRow(0,rootRow.data());
        IVM->setDirection(0,rootDir);
    }else{
        //first line of Matrix
        IVM->setRow(0,varOrder);
        IVM->decodeIVM();

        //compute children bounds (of IVM->node), choose Branching and modify IVM accordingly
        pbb->sltn->getBest(prune->local_best);
        boundAndKeepSurvivors(IVM->getNode());

        //save first line of matrix (bounded root decomposition)
        rootDir = IVM->getDirection(0);
        int c=0;
        for(auto &i : rootRow)
            i=IVM->getCell(0,c++);

        first = false;

        FILE_LOG(logDEBUG) << " === Root : ["<<rootDir<<"]"<<IVM->getNode()<<"\n";
    }
}


template<typename T>
bool
Intervalbb<T>::initAtInterval(std::vector<int> &pos, std::vector<int> &end)
{
    IVM->setDepth(0);

    // std::vector<int>tmppos(size);
    // std::vector<int>tmpend(size);
    //
    // IVM->getInterval(tmppos.data(),tmpend.data());
    //
    // if(IVM->vectorCompare(tmpend.data(),end.data()) != 0 ||
    //     IVM->vectorCompare(tmppos.data(),pos.data()) < 0)
    // {
    //     std::cout<<"current\n";
    //     IVM->displayVector(tmppos.data());
    //     IVM->displayVector(tmpend.data());
    //     std::cout<<"new\n";
    //     IVM->displayVector(pos.data());
    //     IVM->displayVector(end.data());
    // }

    IVM->setPosition(pos.data());
    IVM->setEnd(end.data());

    if (IVM->beforeEnd()) {
        unfold();
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

//using boundChildren
//BEGIN_END OPTIONAL PARAMETER in boundChildren?
template<typename T>
void Intervalbb<T>::boundAndKeepSurvivors(subproblem& _subpb)
{
    std::vector<std::vector<T>> lb(2,std::vector<T>(size,0));
    std::vector<std::vector<T>> prio(2,std::vector<T>(size,0));

    //a priori choice of branching direction
    auto dir = branch->pre_bound_choice(IVM->getDepth());
    IVM->setDirection(dir);

    if(dir<0){    //if undecided
        // get lower bounds : both directions
        primary_bound->boundChildren(
                _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                lb[Branching::Front].data(),lb[Branching::Back].data(),
                prio[Branching::Front].data(),prio[Branching::Back].data(),this->prune->local_best
            );

        //choose branching direction
        dir = (*branch)(
            lb[Branching::Front].data(),
            lb[Branching::Back].data(),
            IVM->getDepth()
        );
    }else if(dir == Branching::Front){
        // get lower bounds : forward only
        primary_bound->boundChildren(
                _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                lb[Branching::Front].data(),nullptr,
                prio[Branching::Front].data(),nullptr,this->prune->local_best
            );
    }else{
        // get lower bounds : backward only
        primary_bound->boundChildren(
                _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                nullptr,lb[Branching::Back].data(),
                nullptr,prio[Branching::Back].data(),this->prune->local_best
            );
    }

    IVM->setDirection(dir);

    //all
    dir = IVM->getDirection();
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
        boundAndKeepSurvivors(IVM->getNode());
    }

    return (state == 1);
}

template<typename T>
void
Intervalbb<T>::unfold()
{
    assert(IVM->intervalValid());
    assert(IVM->getDepth() == 0);

    pbb->sltn->getBest(prune->local_best);

    while (IVM->getDepth() < size - 2) {
        if (IVM->pruningCellState()) {
            IVM->goRight();
            IVM->alignLeft();
            break;
        }

        IVM->incrDepth();
        IVM->generateLine(IVM->getDepth(), false);
        IVM->decodeIVM();

        boundAndKeepSurvivors(IVM->getNode());
    }
} // matrix::unfold


template<typename T>
bool
Intervalbb<T>::boundLeaf(subproblem& node)
{
    FILE_LOG(logDEBUG) << " === bound Leaf"<<std::flush;

    bool better=false;
    int cost=primary_bound->evalSolution(node.schedule.data());

    // std::cout<<cost<<"\t"<<prune->local_best<<"\n";

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
            FILE_LOG(logINFO) << tmp;
            std::cout<<"New Best:\t";
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
