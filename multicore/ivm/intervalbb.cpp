#include <assert.h>

#include "../../common/include/macros.h"

#include "log.h"
#include "pbab.h"
#include "intervalbb.h"



template<typename T>
Intervalbb<T>::Intervalbb(pbab *_pbb) : MCbb<T>(_pbb),first(true), pbb(_pbb), size(_pbb->size),_IVM(std::make_shared<IVM>(size)) //,count_leaves(0),count_decomposed(0)
{
    rootRow = std::vector<T>(size,0);
}

template<typename T>
void
Intervalbb<T>::clear()
{
    _IVM->clearInterval();
}

template<typename T>
void
Intervalbb<T>::setRoot(const int *varOrder)
{
    // _IVM->clearInterval();
    _IVM->setDepth(0);

    if(!first){
        //row 0 and direction have been saved
        _IVM->setRow(0,rootRow.data());
        _IVM->setDirection(0,rootDir);
    }else{
        //first line of Matrix
        _IVM->setRow(0,varOrder);
        _IVM->decodeIVM();

        //compute children bounds (of _IVM->node), choose Branching and modify _IVM accordingly
        pbb->best_found.getBest(this->prune->local_best);
        boundAndKeepSurvivors(_IVM->getNode());

        //save first line of matrix (bounded root decomposition)
        rootDir = _IVM->getDirection(0);
        int c=0;
        for(auto &i : rootRow)
            i=_IVM->getCell(0,c++);

        first = false;

        FILE_LOG(logDEBUG) << " === Root : ["<<rootDir<<"]"<<_IVM->getNode()<<"\n";
    }
}

template<typename T>
void
Intervalbb<T>::setRoot(const std::vector<int> varOrder)
{
    setRoot(varOrder.data());
}



template<typename T>
bool
Intervalbb<T>::initAtInterval(std::vector<int> &pos, std::vector<int> &end)
{
    _IVM->setDepth(0);

    // std::vector<int>tmppos(size);
    // std::vector<int>tmpend(size);
    //
    // _IVM->getInterval(tmppos.data(),tmpend.data());
    //
    // if(_IVM->vectorCompare(tmpend.data(),end.data()) != 0 ||
    //     _IVM->vectorCompare(tmppos.data(),pos.data()) < 0)
    // {
    //     std::cout<<"current\n";
    //     _IVM->displayVector(tmppos.data());
    //     _IVM->displayVector(tmpend.data());
    //     std::cout<<"new\n";
    //     _IVM->displayVector(pos.data());
    //     _IVM->displayVector(end.data());
    // }

    _IVM->setPosition(pos.data());
    _IVM->setEnd(end.data());

    if (_IVM->beforeEnd()) {
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
    auto dir = this->branch->pre_bound_choice(_IVM->getDepth());

    if(dir<0){    //if undecided
        // get lower bounds : both directions
        this->primary_bound->boundChildren(
                _subpb.schedule,_subpb.limit1,_subpb.limit2,
                costFwd.data(),costBwd.data(),
                prioFwd.data(),prioBwd.data(),this->prune->local_best
            );

        //choose branching direction
        dir = (*this->branch)(
            costFwd.data(),
            costBwd.data(),
            _IVM->getDepth()
        );
    }else if(dir == Branching::Front){
        // get lower bounds : forward only
        this->primary_bound->boundChildren(
                _subpb.schedule,_subpb.limit1,_subpb.limit2,
                costFwd.data(),nullptr,
                prioFwd.data(),nullptr,this->prune->local_best
            );
    }else{
        // get lower bounds : backward only
        this->primary_bound->boundChildren(
                _subpb.schedule,_subpb.limit1,_subpb.limit2,
                nullptr,costBwd.data(),
                nullptr,prioBwd.data(),this->prune->local_best
            );
    }

    _IVM->setDirection(dir);

    //all
    // dir = _IVM->getDirection();
    // _IVM->sortSiblingNodes(
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
    if(_IVM->selectNextIt()){ //modify _IVM : set to next subproblem
        _IVM->decodeIVM(); // decode _IVM -> subproblem

        if (_IVM->isLastLine()) {
            this->count_leaves++;
            boundLeaf(_IVM->getNode());
        }else{
            this->count_decomposed++;
            boundAndKeepSurvivors(_IVM->getNode());
        }
        return true;
    }else{
        return false;
    }
}

//initializes _IVM at a given interval
template<typename T>
void
Intervalbb<T>::unfold()
{
    assert(_IVM->intervalValid());
    assert(_IVM->getDepth() == 0);

    pbb->best_found.getBest(this->prune->local_best);

    while (_IVM->getDepth() < size - 2) {
        if (_IVM->pruningCellState()) {
            _IVM->goRight();
            _IVM->alignLeft();
            break;
        }

        _IVM->incrDepth();
        _IVM->generateLine(_IVM->getDepth(), false);
        _IVM->decodeIVM();

        boundAndKeepSurvivors(_IVM->getNode());
    }
} // matrix::unfold


template<typename T>
bool
Intervalbb<T>::boundLeaf(subproblem& node)
{
    FILE_LOG(logDEBUG) << " === bound Leaf"<<std::flush;

    bool better=false;
    int cost=this->primary_bound->evalSolution(node.schedule);

    // std::cout<<cost<<"\t"<<prune->local_best<<"\n";
    // std::cout<<this->count_leaves<<" ";
    // node.print();

    if(!(*this->prune)(cost)){
        better=true;

        //update local best...
        this->prune->local_best=cost;
        //...and global best (mutex)
        pbb->best_found.update(node.schedule.data(),cost);
        pbb->best_found.foundAtLeastOneSolution.store(true);
        pbb->best_found.foundNewSolution.store(true);

        //print new best solution
        if(print_new_solutions){
            subproblem tmp(node);
            tmp.ub=cost;
            tmp.lb=cost;
            FILE_LOG(logINFO) << tmp;
            std::cout<<"New Best:\t";
            tmp.print();
        }
    }
    //mark solution as visited
    _IVM->eliminateCurrent();

    return better;
}

template<typename T>
void
Intervalbb<T>::eliminateJobs(std::vector<T> lb)
{
    int _line=_IVM->getDepth();
    int * jm = _IVM->getRowPtr(_line);

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
