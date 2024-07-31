// #include "fastinsertremove.h"
#include "libheuristic.h"
#include "fastneighborhood.h"

#include <memory>
#include <iostream>

//Deroussi et al...
//1D move (parametrized by "pos")
//1) remove job at position "pos"
//2) insert at best other position
//3.1) if improved, accept
//3.2) if not improved, bestRemove (except pos) and bestInsert (elsewhere)
template<typename T>
int fspnhood<T>::fastBREmove(std::vector<int>& perm, int pos)
{
    int cmax0=m->computeHeads(perm, perm.size());
    int cmax1;

    //remove job from position pos...
    int rjob=m->remove(perm,pos);

    m->tabupos->add(pos);//position forbidden
    m->bestInsert(perm, rjob, cmax1); //insert at best other position
    m->tabupos->clear();

    if(cmax0<cmax1){ //if no improvement
        int rjob2;
        //best best other job
        m->tabujobs->add(rjob);
        int pos2=m->bestRemove(perm, rjob2, cmax1);
        m->tabujobs->clear();
        //add at best other position
        m->tabupos->add(pos2);
        m->bestInsert(perm, rjob2, cmax1);
        m->tabupos->clear();
    }

    return cmax1;
}


template<typename T>
int fspnhood<T>::fastBREmove(std::vector<int>& perm, int pos, int l1, int l2)
{
    int cmax0=m->computeHeads(perm, perm.size());
    //remove job from position pos...
    int rjob=m->remove(perm,pos);

    m->tabupos->clear();
    m->tabujobs->clear();

    int cmax1;
    m->tabupos->add(pos);//position
    for(int i=0;i<l1;i++)
    {
        m->tabujobs->add(perm[i]);
        m->tabupos->add(i);
    }
    for(unsigned i=l2;i<perm.size();i++)
    {
        m->tabujobs->add(perm[i]);
        m->tabupos->add(i);
    }
    m->bestInsert(perm, rjob, cmax1); //insert at best other position
    m->tabupos->clear();

    // int cmax2=cmax1;
    if(cmax0<cmax1){ //if no improvement
        int rjob2;
        //best best other job
        m->tabujobs->add(rjob);
        int pos2=m->bestRemove(perm, rjob2, cmax1);
        m->tabujobs->clear();

        m->tabupos->add(pos2);//position
        m->bestInsert(perm, rjob2, cmax1);
        m->tabupos->clear();
    }

    return cmax1;
}

//kI-move parametrized by pos
template<typename T>
int fspnhood<T>::kImove(std::vector<int>& perm,int pos, int kmax)
{
    int cmax0=m->computeHeads(perm, perm.size());
    int cmax1;

    int rjob;

    m->tabupos->clear();
    m->tabujobs->clear();
    //remove job at position pos (and get removed)
    rjob=m->remove(perm, pos);
    //make job and position tabu
    m->tabujobs->add(rjob);
    m->tabupos->add(pos);

    int k=0;
    bool found=false;
    while(!found && k<kmax)
    {
        //find best position to insert removed job (and get resulting makespan)
        m->bestInsert(perm, rjob, cmax1);
        //accept?
        if(cmax1<=cmax0)
        {
            found=true;
            break;
        }
        else
        {
            k++;
            if(k==kmax)break;

            m->tabupos->clear();
            pos=m->bestRemove2(perm, rjob, cmax1);
            m->tabujobs->add(rjob);
            m->tabupos->add(pos);
        }
    }

    return cmax1;
}

template<typename T>
int fspnhood<T>::fastkImove(std::vector<int>& perm,int kmax)
{
    int k=0;

    int cmax0,cmax1;
    int rjob;

    cmax0=m->computeHeads(perm, perm.size());

    m->tabupos->clear();
    m->tabujobs->clear();

    int pos=m->bestRemove2(perm, rjob, cmax1);
    m->tabujobs->add(rjob);
    m->tabupos->add(pos);

    bool found=false;
    while(!found && k<kmax)
    {
        m->bestInsert(perm, rjob, cmax1);

        if(cmax1<cmax0)
        {
            // printf("%d ---> %d\n",cmax0,cmax1);
            found=true;
            break;
        }
        else
        {
            k++;
            if(k==kmax)break;
            m->tabupos->clear();

            pos=m->bestRemove(perm, rjob, cmax1);
            m->tabujobs->add(rjob);
            m->tabupos->add(pos);
        }
    }

    return cmax1;
}

template<typename T>
int fspnhood<T>::fastkImove(std::vector<int>& perm,int kmax,int l1,int l2)
{
    bool found=false;
    int k=0;

    int cmax0,cmax1;
    int rjob;

    cmax0=m->computeHeads(perm, perm.size());
    // printf("start with %d\n",cmax0);

    m->tabupos->clear();
    m->tabujobs->clear();
    for(int i=0;i<l1;i++)
    {
        m->tabujobs->add(perm[i]);
        m->tabupos->add(i);
    }
    for(unsigned i=l2;i<perm.size();i++)
    {
        m->tabujobs->add(perm[i]);
        m->tabupos->add(i);
    }

    int pos=m->bestRemove(perm, rjob, cmax1);
    m->tabujobs->add(rjob);
    m->tabupos->add(pos);

    while(!found && k<kmax)
    {
        m->bestInsert(perm, rjob, cmax1);
        // printf("get %d\n",cmax1);

        if(cmax1<cmax0)
        {
            found=true;
            break;
        }
        else
        {
            k++;
            if(k==kmax)break;
            // m->tabupos->clear();
            m->tabupos->rem(pos);

            pos=m->bestRemove(perm, rjob, cmax1);
            m->tabujobs->add(rjob);
            m->tabupos->add(pos);
        }
    }

    if(found)return cmax1;
    else return cmax0;
}

template class fspnhood<int>;
