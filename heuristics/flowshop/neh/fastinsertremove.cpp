#include <iostream>
#include <float.h>
#include <limits.h>
#include <random>

#include "fastinsertremove.h"

fastInsertRemove::fastInsertRemove(const std::vector<std::vector<int>> p_times,const int N, const int M) : nbJob(N),nbMachines(M),PTM(p_times){
    head  = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));
    tail  = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));
    inser = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));
    sumPT = std::vector<int>(nbJob);

    for (int i = 0; i < nbJob; i++){
        sumPT[i]=0;
        for (int j = 0; j < nbMachines; j++){
            sumPT[i]+=PTM[j][i];
        }
    }

    tabujobs=std::make_unique<forbidden_list>(nbJob);
    tabupos=std::make_unique<forbidden_list>(nbJob);
};

fastInsertRemove::fastInsertRemove(instance_abstract& _instance)
{
    (_instance.data)->seekg(0);
    (_instance.data)->clear();
    *(_instance.data) >> nbJob;
    *(_instance.data) >> nbMachines;

    PTM = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));

    for (int j = 0; j < nbMachines; j++)
        for (int i = 0; i < nbJob; i++)
            *(_instance.data) >> PTM[j][i];

    //transpose might make sense in terms of memory access, but nbJob > nbMachines ...
    //no clear performance difference observed
    head  = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));
    tail  = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));
    inser = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));
    sumPT = std::vector<int>(nbJob);

    for (int i = 0; i < nbJob; i++){
        sumPT[i]=0;
        for (int j = 0; j < nbMachines; j++){
            sumPT[i]+=PTM[j][i];
        }
    }

    tabujobs=std::make_unique<forbidden_list>(nbJob);
    tabupos=std::make_unique<forbidden_list>(nbJob);
}


void fastInsertRemove::reset()
{
    for(int i=0;i<nbMachines;i++){
        std::fill(head[i].begin(),head[i].end(),0);
        std::fill(tail[i].begin(),tail[i].end(),0);
        std::fill(inser[i].begin(),inser[i].end(),0);
    }
}


//returns makespan of (partial) schedule [ perm[0],perm[1],...perm[len-1] ]
//keeps all completion times in heads
int fastInsertRemove::computeHeads(const std::vector<int>& perm, const int len)
{
    //machine 0
    head[0][0]=PTM[0][perm[0]];
    for(int j=1;j<len;j++){
        head[0][j]=head[0][j-1]+PTM[0][perm[j]];
    }
    //job 0
    for(int k=1;k<nbMachines;k++){
        head[k][0]=head[k-1][0]+PTM[k][perm[0]];
    }

    for(int j=1;j<len;j++){
        int job=perm[j];
        for(int k=1;k<nbMachines;k++){
            head[k][j]=std::max(head[k-1][j],head[k][j-1])+PTM[k][job];
        }
    }

    return head[nbMachines-1][len-1];
}

// compute tails for partial schedule [ perm[0],perm[1],...perm[len-1] ]
void fastInsertRemove::computeTails(const std::vector<int>& perm, int len)
{
    //#machine M
    tail[nbMachines-1][len-1]=PTM[nbMachines-1][perm[len-1]];
    for(int j=len-2;j>=0;j--){
        tail[nbMachines-1][j]=tail[nbMachines-1][j+1]+PTM[nbMachines-1][perm[j]];
    }
    //#job len-1
    for(int k=nbMachines-2;k>=0;k--){
        tail[k][len-1]=tail[k+1][len-1]+PTM[k][perm[len-1]];
    }

    for(int j=len-2;j>=0;j--){
        int job=perm[j];
        for(int k=nbMachines-2;k>=0;k--){
            tail[k][j]=\
            std::max(tail[k+1][j],tail[k][j+1])+\
            PTM[k][job];
        }
    }
}

void fastInsertRemove::computeInser(const std::vector<int>& perm, int len, int job)
{
//  #insert before (position 1)
    inser[0][0]=PTM[0][job];
    for(int k=1;k<nbMachines;k++){
        inser[k][0]=inser[k-1][0]+PTM[k][job];
    }
//  #insert 2nd pos to last
    for(int j=1;j<=len;j++){
        inser[0][j]=head[0][j-1]+PTM[0][job];
        for(int k=1;k<nbMachines;k++){
            inser[k][j]=std::max(inser[k-1][j],head[k][j-1])+PTM[k][job];
        }
    }
}

//get len+1 makespans obtained by inserting "job" into positions 0,1,...,len of partial permuation of length len
//returns cmax before job insertion
int fastInsertRemove::insertMakespans(const std::vector<int>& perm, int len, int job, std::vector<int>& makespans)
{
    int old_cmax = computeHeads(perm, len);
    computeTails(perm, len);
    computeInser(perm, len, job);

    //for each possible insertion position
    for(int i=0;i<=len;i++){
        int tmp = 0;
        for(int j=0;j<nbMachines;j++){
            tmp=std::max(tmp,inser[j][i]+tail[j][i]);
        }
        makespans[i]=tmp;
    }

    return old_cmax;
}

//get len makespans obtained by removing job at position 0,...,len-1 from partial permutation of length len
//returns cmax before removal
int fastInsertRemove::removeMakespans(const std::vector<int>& perm, int len, std::vector<int>& makespans)
{
    computeHeads(perm, len);
    computeTails(perm, len);

    //remove first job (i=0)
    int maxi=0;
    for(int j=0;j<nbMachines;j++){
        maxi=std::max(maxi,tail[j][1]);
    }
    makespans[0]=maxi;

    for(int i=1;i<len-1;i++){
        maxi=0;
        for(int j=0;j<nbMachines;j++){
            maxi=std::max(maxi,head[j][i-1]+tail[j][i+1]);
        }
        makespans[i]=maxi;
    }

    maxi=0;
    for(int j=0;j<nbMachines;j++){
        maxi=std::max(maxi,head[j][len-2]);
    }
    makespans[len-1]=maxi;

    return head[nbMachines-1][len-1];
}

//insert job at position pos in partial permutation of length len
//--> new length : len+1
void fastInsertRemove::insert(std::vector<int>& perm, int &len, int pos, int job)
{
    if(len>=nbJob){
        for(int i=0;i<nbJob;i++)printf("%d ",perm[i]);
        printf("\n permutation full %d %d\n",len,nbJob); exit(-1);
    }

    perm.insert(
        perm.begin()+pos,
        job
    );

    len++;
}

int fastInsertRemove::remove(std::vector<int>& perm, int &len, const int pos){
    int rjob=perm[pos];

    perm.erase(perm.begin()+pos);

    // for(int p=pos;p<len-1;p++){
    //     perm[p]=perm[p+1];
    // }
    // perm[len-1]=0;
    len--;

    return rjob;
}

//insert in position which gives best cmax
int fastInsertRemove::bestInsert(std::vector<int>& perm, int &len, int job, int &cmax)
{
    std::vector<int>makespans(len+1);

    //makespans obtained when inserting job at positions 0,...,len
    insertMakespans(perm, len, job, makespans);

    int minpos=-1;
    int mini=INT_MAX;
    for(int i=0;i<=len;i++){
        if(tabupos->isTabu(i))continue;

        if(makespans[i]<mini){
            mini=makespans[i];
            minpos=i;
        }
    }

    if(minpos>=0){
        insert(perm,len,minpos,job);
    }
    cmax = mini;

    return minpos;
}

//insert in position which gives best cmax
int fastInsertRemove::bestInsert2(std::vector<int>& perm, int &len, int job, int &cmax)
{
    std::vector<int>makespans(len+1);

    //makespans obtained when inserting job at positions 0,...,len
    int oldcmax=insertMakespans(perm, len, job, makespans);

    std::vector<float> weights;
    for(int i=0; i<=len; ++i) {
        float val=(float)oldcmax/(float)(makespans[i]-oldcmax);
        if(tabupos->isTabu(i))val=0.0;
        weights.push_back(val);
    }

    std::default_random_engine generator;
    std::discrete_distribution<int> d1(weights.begin(), weights.end());

    int number = d1(generator);

    insert(perm,len,number,job);

    cmax = makespans[number];
    return number;
}


//remove least well inserted job from perm
//return removed job in remjob
//return position of removed job
int fastInsertRemove::bestRemove(std::vector<int>& perm,int &len,int& remjob, int &cmax)
{
    std::vector<int>makespans(len);

    int oldcmax=removeMakespans(perm, len, makespans);

    int bestpos=-1;
    float maxi=FLT_MIN;

    for(int i=0;i<len;i++){
        int job=perm[i];

        if(tabujobs->isTabu(job))continue;

        float val=(float)(oldcmax-makespans[i])/(float)sumPT[job];

        if(val>maxi){
            maxi=val;
            bestpos=i;
        }
    }

    remjob=-1;
    if(bestpos>=0){
        remjob=remove(perm,len,bestpos);
    }
    cmax=makespans[bestpos];
    return bestpos;
}


//remove least well inserted job from perm
//return removed job in remjob
//return position of removed job
int fastInsertRemove::bestRemove2(std::vector<int>& perm,int &len,int& remjob, int &cmax)
{
    std::vector<int>makespans(len);
    // int *makespans=(int*)malloc(len*sizeof(int));

    int oldcmax=removeMakespans(perm, len, makespans);

    std::vector<float> weights;
    for(int i=0; i<len; ++i) {
        int job=perm[i];
        float val=(float)(oldcmax-makespans[i])/(float)sumPT[job];
        if(tabujobs->isTabu(job))val=0.0;
        weights.push_back(val*val);
    }
    std::default_random_engine generator;

    std::discrete_distribution<int> d1(weights.begin(), weights.end());

    int number = d1(generator);

    cmax=makespans[number];

    remjob=remove(perm,len,number);

    return number;
}
