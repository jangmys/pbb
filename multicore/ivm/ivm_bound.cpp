#include "../../common/include/macros.h"
#include "../../common/include/pbab.h"
#include "../../common/include/solution.h"
#include "../../common/include/subproblem.h"

#include "libbounds.h"

#include "ivm.h"
#include "ivm_bound.h"

#include "branching.h"

#include "operator_factory.h"

template<typename T>
ivm_bound<T>::ivm_bound(pbab* _pbb) : pbb(_pbb){
    size=pbb->size;

    node=new subproblem(size);

    bound.push_back(std::move(OperatorFactory::createBound(pbb->instance,0)));
    bound.push_back(std::move(OperatorFactory::createBound(pbb->instance,1)));

    branch = OperatorFactory::createBranching(arguments::branchingMode,size,pbb->initialUB);
    prune = OperatorFactory::createPruning(arguments::findAll);

    costsBegin = std::vector<std::vector<T>>(2,std::vector<T>(size,0));
    costsEnd = std::vector<std::vector<T>>(2,std::vector<T>(size,0));

    priorityBegin = std::vector<T>(size,0);
    priorityEnd = std::vector<T>(size,0);

    if(rootRow.size()==0)
        rootRow = std::vector<T>(size,0);
}

template<typename T>
ivm_bound<T>::~ivm_bound()
{
    delete node;
}


//reads IVM and sets current subproblem (ivm_bound::node)
template<typename T>
void
ivm_bound<T>::prepareSchedule(const ivm* IVM)
{
    const int* const jM  = IVM->jobMat;
    const int* const pV  = IVM->posVect;
    int _line = IVM->line;

    node->limit1 = -1;
    node->limit2 = size;

    for (int l = 0; l < _line; l++) {
        int pointed = pV[l];
        int job     = absolute(jM[l * size + pointed]);

        if (IVM->dirVect[l] == 0) {
            node->schedule[++node->limit1] = job;
        } else {
            node->schedule[--node->limit2] = job;
        }
    }
    for (int l = 0; l < size - _line; l++){
        node->schedule[node->limit1 + 1 + l] = absolute(jM[_line * size + l]);
    }
} // prepareSchedule


//compute LB on (left or right) children  of subproblem "node"
//optionally providing :
    //already known bounds
    //current local best (for early stopping of LB calculation)
template<typename T>
void
ivm_bound<T>::computeStrongBounds(const int be){
    int _limit1 = node->limit1 + (be==branching::Front?1:0);
    int _limit2 = node->limit2 - (be==branching::Back?1:0);

    // int _limit1 = node->limit1 + (be==FRONT?1:0);
    // int _limit2 = node->limit2 - (be==BACK?1:0);

    std::vector<T> costsFirst;
    std::vector<T> costsSecond;
    std::vector<T> priority;

    if(be==branching::Front){
        costsFirst = costsBegin[STRONG];
        costsSecond = costsBegin[WEAK];
        priority = priorityBegin;
    }else{    // if(be==BACK)
        costsFirst = costsEnd[STRONG];
        costsSecond = costsEnd[WEAK];
        priority = priorityEnd;
    }

    std::fill(costsFirst.begin(),costsFirst.end(),0);
    std::fill(priority.begin(),priority.end(),0);

    int fillPos = (be==branching::Front?_limit1:_limit2);
    int costs[2];

    //for all unscheduled jobs
    for (int i = node->limit1 + 1; i < node->limit2; i++) {
        int job = node->schedule[i];
        //if not yet pruned
        if(!(*prune)(costsSecond[job])){
            swap(&node->schedule[fillPos], &node->schedule[i]);
            bound[STRONG]->bornes_calculer(node->schedule.data(), _limit1, _limit2, costs, arguments::earlyStopJohnson?prune->local_best:-1);
            costsFirst[job] = costs[0];
            priority[job]=costs[1];
            swap(&node->schedule[fillPos], &node->schedule[i]);
        }
    }

#ifdef DEBUG
    if(be==FRONT){
        for (int i = 0; i < size; i++)
            printf("%d ",costsBegin[STRONG][i]);
        printf("\n");
    }
    if(be==BACK){
        for (int i = 0; i < size; i++)
            printf("%d ",costsEnd[STRONG][i]);
        printf("\n");
    }
#endif
}

template<typename T>
void
ivm_bound<T>::boundNode(const ivm* IVM)
{
    int dir=IVM->dirVect[IVM->line];

    if (dir == 1){
        computeStrongBounds(branching::Back);
    }else if(dir == 0){
        computeStrongBounds(branching::Front);
    }else if(dir == -1){
        // printf("eval BE johnson\n");
        computeStrongBounds(branching::Front);
        computeStrongBounds(branching::Back);
    }else{
        perror("boundNode");exit(-1);
    }
}

template<typename T>
void
ivm_bound<T>::computeWeakBounds()
{
    std::fill(costsBegin[STRONG].begin(),costsBegin[STRONG].end(),0);
    std::fill(costsEnd[STRONG].begin(),costsEnd[STRONG].end(),0);

    bound[WEAK]->boundChildren(node->schedule.data(),node->limit1,node->limit2,costsBegin[WEAK].data(),costsEnd[WEAK].data(),priorityBegin.data(),priorityEnd.data());
}

template<typename T>
bool
ivm_bound<T>::boundLeaf(ivm* IVM)
{
    bool better=false;

    pbb->stats.leaves++;

    int cost;
    if(bound[STRONG])cost=bound[STRONG]->evalSolution(node->schedule.data());
    else cost=bound[WEAK]->evalSolution(node->schedule.data());

    if(!(*prune)(cost)){
        better=true;

        //update local best...
        prune->local_best=cost;
        //...and global best (mutex)
        pbb->sltn->update(node->schedule.data(),cost);
        pbb->foundAtLeastOneSolution.store(true);
        pbb->foundNewSolution.store(true);

        //print new best solution (not for NQUEENS)
        if(arguments::printSolutions){
            solution tmp = solution(size);
            tmp.update(node->schedule.data(),cost);
            tmp.print();

            // IVM->displayVector(IVM->posVect);
            // pbb->sltn->print();
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
ivm_bound<T>::boundRoot(ivm* IVM){
	pbb->sltn->getBest(prune->local_best);

    node->limit1=-1;
    node->limit2=size;

    if(!first){
        int c=0;
        for(auto i : rootRow)
            IVM->jobMat[c++]=i;

        // std::cout<<"jobMat\t";
        // IVM->displayVector(IVM->jobMat);

        // memcpy(IVM->jobMat, rootRow, size*sizeof(int));
        IVM->dirVect[0] = rootDir;
    }else{
        first = false;
        // std::cout<<"FIRST"<<std::endl;

        //first line of Matrix
        for(int i=0; i<size; i++){
            node->schedule[i] = pbb->root_sltn->perm[i];
            IVM->jobMat[i] = pbb->root_sltn->perm[i];
            // node->schedule[i] = i;
            // IVM->jobMat[i] = i;
        }
        IVM->line=0;
        node->limit1=-1;
        node->limit2=size;

        // if(arguments::problem[0]=='f')
        //     strongBoundPrune(IVM);
        // else
            weakBoundPrune(IVM);

        // IVM->displayVector(costsBegin[WEAK]);
        // IVM->displayVector(costsEnd[WEAK]);

        //save first line of matrix (bounded root decomposition)
        rootDir = IVM->dirVect[0];
        int c=0;
        for(auto &i : rootRow)
            i=IVM->jobMat[c++];

        // std::cout<<"rootRow\t";
        // IVM->displayVector(rootRow.data());

        // IVM->displayMatrix();
    }

    std::fill(costsBegin[STRONG].begin(),costsBegin[STRONG].end(),0);
    std::fill(costsEnd[STRONG].begin(),costsEnd[STRONG].end(),0);
}


template<typename T>
int
ivm_bound<T>::eliminateJobs(ivm* IVM, std::vector<T> cost1, std::vector<T> cost2, std::vector<T> prio) //,const int dir)
{
    int _line=IVM->line;

    int * jm = IVM->jobMat + _line * size;

    switch (arguments::sortNodes) {
        case 1://non-decreasing cost1
        {
            jm = IVM->jobMat + _line * size;
            gnomeSortByKeyInc(jm, cost1.data(), 0, size-_line-1);
            break;
        }
        case 2://non-decreasing cost1, break ties by priority (set in chooseChildrenSet)
        {
            jm = IVM->jobMat + _line * size;
            gnomeSortByKeysInc(jm, cost1.data(), prio.data(), 0, size-_line-1);
            break;
        }
        case 3:
        {
            jm = IVM->jobMat + _line * size;
            gnomeSortByKeyInc(jm, prio.data(), 0, size-_line-1);
            break;
        }
        case 4:
        {
            jm = IVM->jobMat + _line * size;
            gnomeSortByKeysInc(jm, cost1.data(), prio.data(), 0, size-_line-1);
            break;
        }
    }

    // eliminate
    for (int i = 0; i < size-_line; i++) {
        int job = jm[i];
        if( (*prune)(cost1[job]) || (*prune)(cost2[job]))
        {
            jm[i] = negative(job);
        }
    }

    return 0;
}

template<typename T>
void
ivm_bound<T>::completeSchedule(const int job,const int order)
{
    int* fixedJobs = new int[size];
    memset(fixedJobs,0,size*sizeof(int));

    fixedJobs[job]=1;
    int i,j;

    for(i=0;i<=node->limit1;i++){
        j=node->schedule[i];
        fixedJobs[j]=1;
    }
    for(i=node->limit2;i<size;i++){
        j=node->schedule[i];
        fixedJobs[j]=1;
    }

    //==============
    if(order == branching::Front)
    {
        i=node->limit1+1;

        node->schedule[i++]=job;
        for(int k=0; k<size; k++){
            j=pbb->root_sltn->perm[k];
            if(fixedJobs[j]==1)continue;
            node->schedule[i++]=j;
        }
    }
    if(order == branching::Back)
    {
        i=node->limit1+1;

        for(int k=0; k<size; k++){
            j=pbb->root_sltn->perm[k];
            if(fixedJobs[j]==1)continue;
            node->schedule[i++]=j;
        }
        node->schedule[i++]=job;
    }
    delete[] fixedJobs;
}

template<typename T>
void
ivm_bound<T>::sortSiblingNodes(ivm* IVM)
{
    int _line=IVM->line;

    switch (arguments::sortNodes) {
        case 0:
        {
            int *jm = IVM->jobMat + _line * size;
            int prev_dir=(_line>0)?IVM->dirVect[_line-1]:0;
            if(prev_dir!=IVM->dirVect[_line])
            {
                // std::cout<<"line "<<_line<<" dir "<<IVM->dirVect[_line]<<" reverse\n";
                int i1=0;
                int i2=size-_line-1;
                while(i1<i2){
                    swap(&jm[i1], &jm[i2]);
                    i1++; i2--;
                }
            }
            if(prev_dir==1 && IVM->dirVect[_line]==0){
                for (int l = 0; l < size - _line; l++){
                    node->schedule[node->limit1 + 1 + l] = absolute(jm[l]);
                }
            }
            break;
        }
    }
}

template<typename T>
void
ivm_bound<T>::applyPruning(ivm* IVM, const int first, const int second)
{
    int _line=IVM->line;
    int* jm;// = IVM->jobMat + _line * size;

    if (IVM->dirVect[IVM->line] == 0){
        eliminateJobs(IVM, costsBegin[first],costsBegin[second],priorityBegin);
    }else if(IVM->dirVect[IVM->line] == 1){
        eliminateJobs(IVM, costsEnd[first],costsEnd[second],priorityEnd);
    }
}




/***

getters, setters, ...

***/






template<typename T>
void
ivm_bound<T>::getSchedule(int *sch)
{
    for (int i = 0; i < size; i++) {
        sch[i]=node->schedule[i];
    }
}




template<typename T>
void
ivm_bound<T>::weakBoundPrune(ivm* IVM){
    std::fill(costsBegin[STRONG].begin(),costsBegin[STRONG].end(),0);
    std::fill(costsEnd[STRONG].begin(),costsEnd[STRONG].end(),0);

    // memset(costsBegin[STRONG], 0, size*sizeof(int));
    // memset(costsEnd[STRONG], 0, size*sizeof(int));

    //get lower bounds
    bound[WEAK]->boundChildren(node->schedule.data(),node->limit1,node->limit2,costsBegin[WEAK].data(),costsEnd[WEAK].data(),priorityBegin.data(),priorityEnd.data());

    //make branching decision
    IVM->dirVect[IVM->line] = (*branch)(costsBegin[WEAK].data(),costsEnd[WEAK].data(),IVM->line);

    sortSiblingNodes(IVM);

    applyPruning(IVM,WEAK,WEAK);

    // IVM->displayVector(&IVM->jobMat[IVM->line*size]);
    // std::cout<<*node<<"\n";
}

template<typename T>
void
ivm_bound<T>::mixedBoundPrune(ivm* IVM){
    std::fill(costsBegin[STRONG].begin(),costsBegin[STRONG].end(),0);
    std::fill(costsEnd[STRONG].begin(),costsEnd[STRONG].end(),0);

    // memset(costsBegin[STRONG], 0, size*sizeof(int));
    // memset(costsEnd[STRONG], 0, size*sizeof(int));

    bound[WEAK]->boundChildren(node->schedule.data(),node->limit1,node->limit2,costsBegin[WEAK].data(),costsEnd[WEAK].data(),priorityBegin.data(),priorityEnd.data());

    IVM->dirVect[IVM->line]=(*branch)(costsBegin[WEAK].data(),costsEnd[WEAK].data(),IVM->line);

    boundNode(IVM);

    sortSiblingNodes(IVM);

    applyPruning(IVM,WEAK,STRONG);
}

template<typename T>
void
ivm_bound<T>::strongBoundPrune(ivm* IVM){
    IVM->dirVect[IVM->line]=-1;
    std::fill(costsBegin[WEAK].begin(),costsBegin[WEAK].end(),0);
    std::fill(costsEnd[WEAK].begin(),costsEnd[WEAK].end(),0);
    // memset(costsBegin[WEAK],INT_MAX,size*sizeof(int));
    // memset(costsEnd[WEAK],INT_MAX,size*sizeof(int));
    boundNode(IVM);
    IVM->dirVect[IVM->line]=(*branch)(costsBegin[STRONG].data(),costsEnd[STRONG].data(),IVM->line);

    sortSiblingNodes(IVM);

    applyPruning(IVM,STRONG,STRONG);
}

//explicit instantiations
template class ivm_bound<int>;
// template class ivm_bound<float>;
// template class ivm_bound<double>;
