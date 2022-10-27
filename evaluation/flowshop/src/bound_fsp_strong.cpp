#include <limits.h>
#include <iostream>
#include <algorithm>
//
// #include "c_bound_simple.h"
// #include "c_bound_johnson.h"

#include "bound_fsp_strong.h"

// lower bound and evaluation function for the PFSP
//
// m-machine bound bound as in ... Lageweg et al


// ==============================================================
// INITIALIZATIONS
// ==============================================================
void
bound_fsp_strong::init(instance_abstract * _instance)
{
    pthread_mutex_lock(&_instance->mutex_instance_data);
    (_instance->data)->seekg(0);
    (_instance->data)->clear();
    *(_instance->data) >> nbJob;
    *(_instance->data) >> nbMachines;

    data_lb1 = new_bound_data(nbJob,nbMachines);
    data_lb2 = new_johnson_bd_data(data_lb1);

    // allocate(nbJob,nbMachines);

    fill_machine_pairs(data_lb2);

    nbMachinePairs = data_lb2->nb_machine_pairs;
    // nbMachinePairs = nbMachines * (nbMachines - 1) / 2;

    // read matrix of processing times from instance-data (stringstream)
    PTM = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));
    p_times = std::vector<int>(nbJob*nbMachines);

    for (int j = 0; j < nbMachines; j++){
        for (int i = 0; i < nbJob; i++){
            *(_instance->data) >> PTM[j][i];
            p_times[j*nbJob+i] = PTM[j][i];
            data_lb1->p_times[j*nbJob+i] = PTM[j][i];
        }
    }
    pthread_mutex_unlock(&_instance->mutex_instance_data);


    // fill auxiliary data for LB computation
    // [to be called in this order!!]
    fill_min_heads_tails(data_lb1,data_lb1->min_heads,data_lb1->min_tails);

    fillMachinePairs();
    fillLags();
    fillJohnsonSchedules();

    rewards = (int *) malloc(nbMachinePairs * sizeof(int));
    countMachinePairs = (int *) malloc(nbMachinePairs * sizeof(int));
    machinePairOrder  = (int *) malloc(nbMachinePairs * sizeof(int));

    for (int i = 0; i < nbMachinePairs; i++) {
        rewards[i] = 0;
        countMachinePairs[i] = 0;
        machinePairOrder[i]  = i;
    }

    flag   = std::vector<int>(nbJob);
    front  = std::vector<int>(nbMachines);
    back  = std::vector<int>(nbMachines);
    remain  = std::vector<int>(nbMachines);
}

// void
// bound_fsp_strong::fillMinHeadsTails()
// {
//     fill_min_heads_tails(data_lb1,data_lb1->min_heads,data_lb1->min_tails);
// }

void
bound_fsp_strong::fillMachinePairs()
{
    // [0 0 0 ...  0  1 1 1 ... ... M-3 M-3 M-2 ]
    // [1 2 3 ... M-1 2 3 4 ... ... M-2 M-1 M-1 ]
    for(int i=0; i<nbMachines-1; i++){
        for(int j=i+1; j<nbMachines; j++){
            machine_pairs.push_back(std::make_pair(i,j));
        }
    }
}

void
bound_fsp_strong::fillLags()
{
    lags = std::vector<std::vector<int> >(nbMachinePairs, std::vector<int>(nbJob,0));
    p_lags = std::vector<int>(nbMachinePairs*nbJob,0);

    // int *tmpLags = new int[nbMachinePairs*nbJob];

    fill_lags(data_lb1->p_times,nbJob,nbMachines,p_lags.data());

    fill_lags(data_lb1->p_times,nbJob,nbMachines,data_lb2->lags);

    for (int i = 0; i < nbMachinePairs; i++) {
        for (int j = 0; j < nbJob; j++) {
            lags[i][j] = p_lags[i*nbJob + j];
        }
    }
}

void
bound_fsp_strong::fillJohnsonSchedules()
{
    std::vector<int>tmpJohnson(nbMachinePairs*nbJob);

    fill_johnson_schedules(p_times.data(),p_lags.data(),nbJob,nbMachines,tmpJohnson.data());

    for(int s=0; s<nbMachinePairs; s++){
        std::vector<int> johnson_seq(nbJob);
        for (int i = 0; i < nbJob; i++) {
            johnson_seq[i] = tmpJohnson[s*nbJob+i];
        }
        johnson_schedules.push_back(johnson_seq);
    }

    fill_johnson_schedules(p_times.data(),p_lags.data(),nbJob,nbMachines,data_lb2->johnson_schedules);
}

// ==============================================================
// Lower bound computation
// ==============================================================
// initial sequence
void
bound_fsp_strong::scheduleFront(int permutation[], int limite1, int limite2, int * idle)
{
    schedule_front(data_lb1->p_times, data_lb1->min_heads, nbJob, nbMachines, permutation,limite1,front.data());
    // schedule_front(data_lb1, permutation,limite1,front.data());
}

// reverse problem...
void
bound_fsp_strong::scheduleBack(int permutation[], int limite2, int * idle)
{
    schedule_back(data_lb1->p_times, data_lb1->min_tails, nbJob, nbMachines, permutation,limite2,back.data());
    // schedule_back(data_lb1, permutation,limite2,back.data());
}




void
bound_fsp_strong::configureBound(const int _branchingMode, const int _earlyExit, const int _machinePairs)
{
    branchingMode = _branchingMode;
    earlyExit     = _earlyExit;
    machinePairs  = _machinePairs;
}







inline
void
bound_fsp_strong::initCmax(std::pair<int,int>& tmp, std::pair<int,int>& ma, int ind)
{
    ma.first = machine_pairs[ind].first;
    ma.second = machine_pairs[ind].second;

    tmp.first = front[ma.first];
    tmp.second = front[ma.second];
}

inline
void
bound_fsp_strong::cmaxFin(std::pair<int,int>& tmp, std::pair<int,int> ma)
{
    if (tmp.second + back[ma.second] > tmp.first + back[ma.first])
        tmp.second = tmp.second + back[ma.second];
    else
        tmp.second = tmp.first + back[ma.first];
}

inline
void
bound_fsp_strong::heuristiqueCmax(int *flag, std::pair<int,int>& tmp, std::pair<int,int> ma, int ind)
{
    int tmp0 = tmp.first;
    int tmp1 = tmp.second;
    int ma0  = ma.first;
    int ma1  = ma.second;

    int test = compute_cmax_johnson(data_lb1,data_lb2,flag,&tmp0,&tmp1,ma.first,ma.second,ind);

    tmp.first = tmp0;
    tmp.second = tmp1;
    // for (int j = 0; j < nbJob; j++) {
    //     int jobCour = johnson_schedules[ind][j];
    //     // j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
    //     if (flag[jobCour] == 0) {
    //         // add jobCour to ma0 and ma1
    //         tmp0 += PTM[ma0][jobCour];
    //         if (tmp1 > tmp0 + lags[ind][jobCour])
    //             tmp1 += PTM[ma1][jobCour];
    //         else
    //             tmp1 = tmp0 + lags[ind][jobCour] + PTM[ma1][jobCour];
    //     }
    // }

    // if(test != tmp.second){
    //     printf("%d %d\n",test,tmp.second);
    // }

}

int
bound_fsp_strong::borneInfMakespan(int * valBorneInf, int minCmax)
{
    int moinsBon = 0;
    std::pair<int,int> ma;
    // int ma[2];  /*Contient les rang des deux machines considere.*/
    std::pair<int,int> tmp; /*Contient les temps sur les machines considere*/

    int bestind = 0;

    // for all machine-pairs : O(m^2) m*(m-1)/2
    for (int l = 0; l < nbMachinePairs; l++) {
        // start by most successful machine-pair....only useful if early exit allowed
        int i = machinePairOrder[l];
        // add min start times and get ma[0],ma[1] from index i
        // initCmax(tmp, ma, i);

        ma.first = machine_pairs[i].first;
        ma.second = machine_pairs[i].second;

        // (1,m),(2,m),...,(m-1,m)
        if (machinePairs == 1 && ma.second < nbMachines - 1) continue;
        // (1,2),(2,3),...,(m-1,m)
        if (machinePairs == 2 && ma.second != ma.first + 1) continue;

        tmp.first = front[ma.first];
        tmp.second = front[ma.second];

        // compute cost for johnson sequence : O(n)
        heuristiqueCmax(flag.data(),tmp, ma, i);
        // complete bound with scheduled cost
        cmaxFin(tmp, ma);

        // take max
        if (tmp.second > moinsBon) {
            // best index
            bestind = i;
            // update max
            moinsBon = tmp.second;
        }

        // early exit from johnson (if lb > best)
        if (moinsBon > minCmax && minCmax != -1) {
            valBorneInf[0] = moinsBon;
            break;
        }
    }

    // ++ successful machine pair
    countMachinePairs[bestind]++;

    valBorneInf[0] = moinsBon;
    return bestind;
} // borneInfMakespan

int
bound_fsp_strong::borneInfLearn(int * valBorneInf, int UB, bool earlyExit)
{
    // reset periodically...
    if (nbbounds > 100 * 2 * nbJob) {
        nbbounds = 0;
        for (int k = 0; k < nbMachinePairs; k++) {
            rewards[k] = 0;///=100;
        }
    }

    int maxLB = 0;
    std::pair<int,int> ma;
    std::pair<int,int> tmp; /*Contient les temps sur les machines considere*/

    int i, j, l;
    int bestind = 0;

    i = 1, j = 2;
    while (i < nbMachinePairs) {
        if (rewards[machinePairOrder[i - 1]] < rewards[machinePairOrder[i]]) {
            std::swap(machinePairOrder[i - 1], machinePairOrder[i]);
            if ((--i)) continue;
        }
        i = j++;
    }

    // restrict to best nbMachines
    int nbPairs = nbMachines;
    // learn...
    if (nbbounds < 2 * nbJob) nbPairs = nbMachinePairs;
    nbbounds++;

    for (l = 0; l < nbPairs; l++) {
        // start by most successful machine-pair....only useful if early exit allowed
        i = machinePairOrder[l];

        initCmax(tmp, ma, i);// add min start times

        heuristiqueCmax(flag.data(),tmp, ma, i);// compute johnson sequence //O(n)
        cmaxFin(tmp, ma);

        if (tmp.second > maxLB) {
            bestind = i;
            maxLB   = tmp.second;
        }
        if (earlyExit && (maxLB > UB) && (nbPairs < nbMachinePairs)) {
            break;
        }
    }

    rewards[bestind]++;

    valBorneInf[0] = maxLB;
    return bestind;
} // bound_fsp_strong::borneInfLearn


// ==============================
// COMPUTE BOUND
// ==============================
int
bound_fsp_strong::calculBorne(int minCmax)
{
    int valBorneInf[2];

    if (machinePairs == 3) {
        borneInfLearn(valBorneInf, minCmax, true);
        return valBorneInf[0];
    } else {
        return lb_makespan(data_lb1,data_lb2,flag.data(),front.data(),back.data(),minCmax);
        // valBorneInf[0]=borneInfMakespan(valBorneInf, minCmax);
    }

}



void
bound_fsp_strong::setFlags(int permutation[], int limite1, int limite2, int *flag)
{
    for(int i=0;i<nbJob;i++)
        flag[i]=0;
    for (int j = 0; j <= limite1; j++)
        flag[permutation[j]] = -1;
    for (int j = limite2; j < nbJob; j++)
        flag[permutation[j]] = permutation[j] + 1;
}


void
bound_fsp_strong::bornes_calculer(int permutation[], int limite1, int limite2, int * couts, int best)
{
    if (limite2 - limite1 == 1) {
        // printf("this happens\n");
        couts[0] = evalSolution(permutation);
        //        couts[1]=couts[0];
    } else {
        setFlags(permutation, limite1, limite2,flag.data());

        // set_nombres(limite1, limite2);
        // compute front
        couts[1] = 0;
        scheduleFront(permutation, limite1, limite2, &couts[1]);

        // compute tail
        scheduleBack(permutation, limite2, &couts[1]);

        couts[0] = calculBorne(best);
        // std::cout<<"LB "<<best<<" "<<couts[0]<<"\n";
    }
}

void
bound_fsp_strong::boundChildren(int permutation[], int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best)
{
    std::vector<int>costs(2,0);

    for (int i = limit1 + 1; i < limit2; i++) {
        int job = permutation[i];

        //front
        if(costsBegin){
            std::swap(permutation[limit1 + 1], permutation[i]);
            bornes_calculer(permutation, limit1 + 1, limit2, costs.data(), best);
            costsBegin[job] = costs[0];
            prioBegin[job]=costs[1];
            std::swap(permutation[limit1 + 1], permutation[i]);
        }
        //back
        if(costsEnd){
            std::swap(permutation[limit2 - 1], permutation[i]);
            bornes_calculer(permutation, limit1, limit2 - 1, costs.data(), best);
            costsEnd[job] = costs[0];
            prioEnd[job]=costs[1];
            std::swap(permutation[limit2 - 1], permutation[i]);
        }
    }
}


int
bound_fsp_strong::evalSolution(int * permut)
{
    std::vector<int> tmp(nbMachines,0);

    for (int i = 0; i < nbJob; i++) {
        int job = permut[i];
        tmp[0] += PTM[0][job];
        for (int j = 1; j < nbMachines; j++) {
            tmp[j] = std::max(tmp[j - 1], tmp[j]) + PTM[j][job];
        }
    }

    return tmp[nbMachines - 1];
}

int
bound_fsp_strong::bornes_calculer(int * schedule, int limit1, int limit2)
{
    return 0;
    // bornes_calculer(p.permutation, p.limite1, p.limite2,p.couts,999999);
    // p.couts_nbMachinePairs=p.couts[0]+p.couts[1];
}
