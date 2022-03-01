#include <limits.h>
#include <iostream>
#include <algorithm>

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
    (_instance->data)->seekg(0);
    (_instance->data)->clear();
    *(_instance->data) >> nbJob;
    *(_instance->data) >> nbMachines;

    nbMachinePairs = nbMachines * (nbMachines - 1) / 2;

    // read matrix of processing times from instance-data (stringstream)
    PTM = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));

    for (int j = 0; j < nbMachines; j++)
        for (int i = 0; i < nbJob; i++)
            *(_instance->data) >> PTM[j][i];

    // fill auxiliary data for LB computation
    // [to be called in this order!!]
    fillMinHeadsTails();
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

void
bound_fsp_strong::fillMinHeadsTails()
{
    min_heads = std::vector<int>(nbMachines);
    min_tails = std::vector<int>(nbMachines);

    // 1/ min start times on each machine
    for (int k = 0; k < nbMachines; k++) min_heads[k] = INT_MAX;
    // per definition =0 on first machine
    min_heads[0] = 0;

    for (int i = 0; i < nbJob; i++) {
        std::vector<int> tmp(nbMachines, 0);

        tmp[0] += PTM[0][i];
        for (int k = 1; k < nbMachines; k++) {
            tmp[k] = tmp[k - 1] + PTM[k][i];
        }
        for (int k = 1; k < nbMachines; k++) {
            min_heads[k] = (tmp[k - 1] < min_heads[k]) ? tmp[k - 1] : min_heads[k];
        }
    }



    // 2/ min run-out times on each machine
    std::fill(min_tails.begin(), min_tails.end(), INT_MAX);
    // per definition =0 on last machine
    min_tails[nbMachines - 1] = 0;

    for (int i = 0; i < nbJob; i++) {
        std::vector<int> tmp(nbMachines, 0);

        tmp[nbMachines - 1] += PTM[nbMachines - 1][i];
        for (int k = nbMachines - 2; k >= 0; k--) {
            tmp[k] = tmp[k + 1] + PTM[k][i];
        }
        for (int k = nbMachines - 2; k >= 0; k--) {
            min_tails[k] = std::min(tmp[k + 1], min_tails[k]);
        }
    }
}

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

    // for all jobs and all machine-pairs
    for (int i = 0; i < nbMachinePairs; i++) {
        int m1 = machine_pairs[i].first;
        int m2 = machine_pairs[i].second;

        for (int j = 0; j < nbJob; j++) {
            lags[i][j] = 0;

            // term q_iuv in [Lageweg'78]
            for (int k = m1 + 1; k < m2; k++){
                lags[i][j] += PTM[k][j];
            }
        }
    }
}

void
bound_fsp_strong::fillJohnsonSchedules()
{
    //for each machine-pair (m1,m2), solve 2-machine FSP with processing times
    //  p_1i = PTM[m1][i] + lags[s][i]
    //  p_2i = PTM[m2][i] + lags[s][i]
    //using Johnson's algorithm [Johnson, S. M. (1954). Optimal two-and three-stage production schedules with setup times included.closed access Naval research logistics quarterly, 1(1), 61–68.]
    for(int s=0; s<nbMachinePairs; s++){
        int m1 = machine_pairs[s].first;
        int m2 = machine_pairs[s].second;

        //partition into 2 sets {j|p_1j < p_2j} and {j|p_1j >= p_2j}
        std::vector<std::pair<int,int>> set1;
        std::vector<std::pair<int,int>> set2;
        for (int i = 0; i < nbJob; i++) {
            if (PTM[m1][i] < PTM[m2][i]) {
                set1.push_back(std::make_pair(i,PTM[m1][i] + lags[s][i]));
            }else{
                set2.push_back(std::make_pair(i,PTM[m2][i] + lags[s][i]));
            }
        }

        //sort set1 in increasing order of p_1j
        std::sort(set1.begin(),set1.end(),
            [](std::pair<int,int> a, std::pair<int,int> b)
            {return a.second < b.second; }
        );
        //sort set2 in decreasing order of p_2j
        std::sort(set2.begin(),set2.end(),
            [](std::pair<int,int> a, std::pair<int,int> b)
            {return a.second > b.second; }
        );

        //[set1,set2] form an optimal sequence
        std::vector<int> johnson_seq;
        for(auto &i : set1)
            johnson_seq.push_back(i.first);
        for(auto &i : set2)
            johnson_seq.push_back(i.first);

        johnson_schedules.push_back(johnson_seq);
    }

    // for(auto& p : johnson_schedules)
    // {
    //     for(auto &i : p)
    //     {
    //         std::cout<<i<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
}

// ==============================================================
// Lower bound computation
// ==============================================================
// initial sequence
void
bound_fsp_strong::scheduleFront(int permutation[], int limite1, int limite2, int * idle)
{
    if (limite1 == -1) {
        for (int i = 0; i < nbMachines; i++)
            front[i] = min_heads[i];
        *idle = 0;
        return;
    }
    std::fill(front.begin(),front.end(),0);

    for (int j = 0; j <= limite1; j++) {
        int job      = permutation[j];
        front[0] = front[0] + PTM[0][job];
        for (int m = 1; m < nbMachines; m++) {
            *idle   += std::max(0, front[m - 1] - front[m]);
            front[m] = std::max(front[m],
                front[m - 1]) + PTM[m][job];
        }
    }
}

// reverse problem...
void
bound_fsp_strong::scheduleBack(int permutation[], int limite2, int * idle)
{
    if (limite2 == nbJob) {
        for (int i = 0; i < nbMachines; i++)
            back[i] = min_tails[i];
        // memcpy(back, minTempsDep, nbMachines * sizeof(int));
        *idle = 0;
        return;
    }
    std::fill(back.begin(),back.end(),0);

    for (int k = nbJob - 1; k >= limite2; k--) {
        int jobCour = permutation[k];

        back[nbMachines - 1] += PTM[(nbMachines - 1)][jobCour];
        for (int j = nbMachines - 2; j >= 0; j--) {
            *idle  += std::max(0, back[j + 1] - back[j]);
            back[j] = std::max(back[j], back[j + 1]) + PTM[j][jobCour];
        }
    }
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
bound_fsp_strong::heuristiqueCmax(std::pair<int,int>& tmp, std::pair<int,int> ma, int ind)
{
    int tmp0 = tmp.first;
    int tmp1 = tmp.second;
    int ma0  = ma.first;
    int ma1  = ma.second;

    for (int j = 0; j < nbJob; j++) {
        int jobCour = johnson_schedules[ind][j];
        // j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
        if (flag[jobCour] == 0) {
            // add jobCour to ma0 and ma1
            tmp0 += PTM[ma0][jobCour];
            if (tmp1 > tmp0 + lags[ind][jobCour])
                tmp1 += PTM[ma1][jobCour];
            else
                tmp1 = tmp0 + lags[ind][jobCour] + PTM[ma1][jobCour];
        }
    }
    tmp.first = tmp0;
    tmp.second = tmp1;
}

int
bound_fsp_strong::borneInfMakespan(int * valBorneInf, int minCmax)
{
    int moinsBon = 0;
    std::pair<int,int> ma;
    // int ma[2];  /*Contient les rang des deux machines considere.*/
    std::pair<int,int> tmp; /*Contient les temps sur les machines considere*/

    int bestind = 0;

    // sort machine-pairs by success-count (if earlyExit enabled)
    // at most one swap...
    if (earlyExit) {
        int i = 1;
        int j = 2;
        while (i < nbMachinePairs) {
            if (countMachinePairs[machinePairOrder[i - 1]] < countMachinePairs[machinePairOrder[i]]) {
                std::swap(machinePairOrder[i - 1],machinePairOrder[i]);
                if ((--i)) continue;
            }
            i = j++;
        }
    }

    // for all machine-pairs : O(m^2) m*(m-1)/2
    for (int l = 0; l < nbMachinePairs; l++) {
        // start by most successful machine-pair....only useful if early exit allowed
        int i = machinePairOrder[l];
        // add min start times and get ma[0],ma[1] from index i
        initCmax(tmp, ma, i);

        // (1,m),(2,m),...,(m-1,m)
        if (machinePairs == 1 && ma.second < nbMachines - 1) continue;
        // (1,2),(2,3),...,(m-1,m)
        if (machinePairs == 2 && ma.second != ma.first + 1) continue;

        // compute cost for johnson sequence : O(n)
        heuristiqueCmax(tmp, ma, i);
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

        heuristiqueCmax(tmp, ma, i);// compute johnson sequence //O(n)
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
    } else {
        borneInfMakespan(valBorneInf, minCmax);
    }

    return valBorneInf[0];
}



void
bound_fsp_strong::setFlags(int permutation[], int limite1, int limite2)
{
    std::fill(flag.begin(),flag.end(),0);

    for (int j = 0; j <= limite1; j++)
        flag[permutation[j]] = -1;
    for (int j = limite2; j < nbJob; j++)
        flag[permutation[j]] = permutation[j] + 1;
}


void
bound_fsp_strong::bornes_calculer(int permutation[], const int limite1, const int limite2, int * couts, const int best)
{
    if (limite2 - limite1 <= 2) {
        //        printf("this happens\n");
        couts[0] = evalSolution(permutation);
        //        couts[1]=couts[0];
    } else {
        setFlags(permutation, limite1, limite2);

        // set_nombres(limite1, limite2);
        // compute front
        couts[1] = 0;
        scheduleFront(permutation, limite1, limite2, &couts[1]);

        // compute tail
        scheduleBack(permutation, limite2, &couts[1]);

        couts[0] = calculBorne(best);
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

void
bound_fsp_strong::bornes_calculer(int * schedule, int limit1, int limit2)
{
    // bornes_calculer(p.permutation, p.limite1, p.limite2,p.couts,999999);
    // p.couts_nbMachinePairs=p.couts[0]+p.couts[1];
}
