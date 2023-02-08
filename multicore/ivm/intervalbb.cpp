#include <assert.h>

#include "../../common/include/macros.h"

#include "log.h"
#include "pbab.h"
#include "intervalbb.h"



template<typename T>
Intervalbb<T>::Intervalbb(pbab *_pbb) : MCbb<T>(_pbb),first(true), pbb(_pbb), size(_pbb->size),IVM(std::make_shared<ivm>(size)) //,count_leaves(0),count_decomposed(0)
{
    rootRow = std::vector<T>(size,0);
}

template<typename T>
void
Intervalbb<T>::clear()
{
    IVM->clearInterval();
}

template<typename T>
void
Intervalbb<T>::setRoot(const int *varOrder)
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
        pbb->best_found.getBest(this->prune->local_best);
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
    this->prune->local_best = bestCost;
}

template<typename T>
void Intervalbb<T>::run()
{
    while(next());

    pbb->stats.totDecomposed += this->count_decomposed;
    pbb->stats.leaves += this->count_leaves;
}

//using boundChildren
//BEGIN_END OPTIONAL PARAMETER in boundChildren?
template<typename T>
void Intervalbb<T>::boundAndKeepSurvivors(subproblem& _subpb)
{
    std::vector<T> costFwd(size,0);
    std::vector<T> costBwd(size,0);

    std::vector<T> prioFwd(size,0);
    std::vector<T> prioBwd(size,0);

    //a priori choice of branching direction
    auto dir = this->branch->pre_bound_choice(IVM->getDepth());

    if(dir<0){    //if undecided
        // get lower bounds : both directions
        this->primary_bound->boundChildren(
                _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                costFwd.data(),costBwd.data(),
                prioFwd.data(),prioBwd.data(),this->prune->local_best
            );

        //choose branching direction
        dir = (*this->branch)(
            costFwd.data(),
            costBwd.data(),
            IVM->getDepth()
        );
    }else if(dir == Branching::Front){
        // get lower bounds : forward only
        this->primary_bound->boundChildren(
                _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                costFwd.data(),nullptr,
                prioFwd.data(),nullptr,this->prune->local_best
            );
    }else{
        // get lower bounds : backward only
        this->primary_bound->boundChildren(
                _subpb.schedule.data(),_subpb.limit1,_subpb.limit2,
                nullptr,costBwd.data(),
                nullptr,prioBwd.data(),this->prune->local_best
            );
    }

    IVM->setDirection(dir);

    //all
    // dir = IVM->getDirection();
    // IVM->sortSiblingNodes(
    //     lb[dir],
    //     prio[dir]
    // );

    if(dir==Branching::Front)
        eliminateJobs(costFwd);
    else
        eliminateJobs(costBwd);

}



template<typename T>
bool Intervalbb<T>::next()
{
    if(IVM->selectNextIt()){ //modify IVM : set to next subproblem
        IVM->decodeIVM(); // decode IVM -> subproblem

        if (IVM->isLastLine()) {
            this->count_leaves++;
            boundLeaf(IVM->getNode());
        }else{
            this->count_decomposed++;
            boundAndKeepSurvivors(IVM->getNode());
        }
        return true;
    }else{
        return false;
    }
}

//initializes IVM at a given interval
template<typename T>
void
Intervalbb<T>::unfold()
{
    assert(IVM->intervalValid());
    assert(IVM->getDepth() == 0);

    pbb->best_found.getBest(this->prune->local_best);

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
    int cost=this->primary_bound->evalSolution(node.schedule.data());

    // std::cout<<cost<<"\t"<<prune->local_best<<"\n";

    if(!(*this->prune)(cost)){
        better=true;

        //update local best...
        this->prune->local_best=cost;
        //...and global best (mutex)
        pbb->best_found.update(node.schedule.data(),cost);
        pbb->best_found.foundAtLeastOneSolution.store(true);
        pbb->best_found.foundNewSolution.store(true);

        //print new best solution
        if(arguments::printSolutions){
            subproblem tmp(node);
            tmp.set_fitness(cost);
            tmp.set_lower_bound(cost);
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
        if( (*this->prune)(lb[job]) ){
            jm[i] = negative(job);
        }
    }
}

template class MCbb<int>;
template class Intervalbb<int>;
