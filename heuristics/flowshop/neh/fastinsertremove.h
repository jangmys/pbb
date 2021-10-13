#ifndef FASTINSERTREMOVE_H_
#define FASTINSERTREMOVE_H_

#include <vector>
#include <iostream>
#include <memory>

#include "libbounds.h"

// #include "../../bounds/include/instance_abstract.h"

struct tabulist
{
    int nmax;
    int num;
    std::vector<int> arr;

    tabulist(int N)
    {
        nmax=N;
        num=0;
        arr=std::vector<int>(nmax);
    }

    bool isTabu(int a){
        for(int i=0;i<num;i++){
            if(arr[i]==a)return true;
        }
        return false;
    };
    void add(int a){
        arr[num++]=a;
    };
    void clear(){
        for(int i=0;i<nmax;i++){
            arr[i]=0;
        }
        num=0;
    };
    void rem(int a){
        int i=num-1;
        while(arr[i]!=a && i>=0)i--;

        if(i<0){
            std::cout<<"can't remove "<<a<<" from tblist["<<num<<"] : not found\n";
            for(int k=0;k<num;k++){
                std::cout<<arr[k]<<" ";
            }
            std::cout<<std::endl;
            exit(-1);
        }
        else{
            for(int j=i;j<num-2;j++)
            {
                arr[j]=arr[j+1];
            }
        }
        num--;
    };
};

template<typename T>
class fastInsertRemove{
public:
    fastInsertRemove(instance_abstract* inst);
    ~fastInsertRemove();
    // instance_abstract * instance;

    int nbJob;
    int nbMachines;

    std::vector<std::vector<int>> PTM;
    std::vector<int> sumPT;

    std::vector<std::vector<int>> head;
    std::vector<std::vector<int>> tail;
    std::vector<std::vector<int>> inser;

    tabulist *tabujobs;
    tabulist *tabupos;

    int insertMakespans(int const* const perm, int len, int job, std::vector<int>& makespans);
    int removeMakespans(int const* const perm, int len, std::vector<int>& makespans);

    int computeHeads(int const* const perm, int len);
    void computeTails(int const* const perm, int len);
    void computeInser(int const* const perm, int len, int job);

    void insert(int* const perm, int &len, int pos, int job);
    int remove(int *perm, int &len, const int pos);

    int bestInsert(int *perm, int &len, int job, int &cmax);
    int bestRemove(int *perm, int &len, int &remjob, int &cmax);

    int bestInsert2(int *perm, int &len, int job, int &cmax);
    int bestRemove2(int *perm, int &len, int &remjob, int &cmax);

    void computeInserBlock(int *perm, int len, int * block, int bl);
    int insertBlockMakespans(int* perm, int len, int* block, int bl, int* makespans);
    void blockInsert(int* perm, int &len, int pos, int*block, int bl);
    int bestBlockInsert(int* perm, int len, int*block, int bl);
};

#endif
