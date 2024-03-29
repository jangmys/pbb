/*
Permutation Flow-Shop

Fast insertion and fast removal (allows to know the best position for inserting a given job (indexed k ) in a partial permutation of k −1 jobs. More precisely, it allows to compute the k makespans obtained by inserting job k at the i^th position (1≤i≤k) in O(km) (as for the makespan calculation). Same for removal

Taillard, E. (1990). Some efficient heuristic methods for the flow shop sequencing problem. European Journal of Operational Research, 47, 67-74.

Laurent Deroussi, Michel Gourgand, Sylvie Norre, New effective neighborhoods for the permutation flow shop problem, Research Report LIMOS/RR-06-09


Author : Jan GMYS (jan.gmys@univ-lille.fr)
*/
#ifndef FASTINSERTREMOVE_H_
#define FASTINSERTREMOVE_H_

#include <vector>
#include <iostream>
#include <memory>

#include "libbounds.h"


struct forbidden_list
{
    std::vector<int>flags;

    explicit forbidden_list(size_t _len) : flags(_len,0) { }

    bool isTabu(int a){
        return flags[a];
    }

    void add(int a){
        flags[a]=1;
    }

    void clear(){
        std::fill(flags.begin(), flags.end(), 0);
    }

    void rem(int a){
        if(!flags[a]){
            std::cout<<"can't remove "<<a<<" : not set\n";
            exit(-1);
        }else{
            flags[a]=0;
        }
    }
};


class fastInsertRemove{
public:
    fastInsertRemove(const std::vector<std::vector<int>> p_times,const int N, const int M);
    fastInsertRemove(instance_abstract& inst);

    unsigned int nbJob;
    int nbMachines;

    std::vector<std::vector<int>> PTM;
    std::vector<int> sumPT;

    std::unique_ptr<forbidden_list> tabujobs;
    std::unique_ptr<forbidden_list> tabupos;

    void reset();

    int insertMakespans(const std::vector<int>& perm, int len, int job, std::vector<int>& makespans);
    int removeMakespans(const std::vector<int>& perm, int len, std::vector<int>& makespans);

    int computeHeads(const std::vector<int>& perm, const int len);
    void computeTails(const std::vector<int>& perm, int len);
    void computeInser(const std::vector<int>& perm, int len, int job);

    void insert(std::vector<int>& perm, int pos, int job);
    int remove(std::vector<int>& perm, const int pos);

    int bestInsert(std::vector<int>& perm, int job, int &cmax);
    int bestRemove(std::vector<int>& perm, int &remjob, int &cmax);
    int bestRemove2(std::vector<int>& perm, int &remjob, int &cmax);

    // int bestInsert2(std::vector<int>& perm, int &len, int job, int &cmax);
private:
    std::vector<std::vector<int>> head;
    std::vector<std::vector<int>> tail;
    std::vector<std::vector<int>> inser;
};

#endif
