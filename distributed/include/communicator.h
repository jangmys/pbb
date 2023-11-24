/*
===================================
Send/Receive routines for work units (sets of intervals)
--------------------------------------------------------
Author : Jan Gmys (jan.gmys@univ-lille.fr)
------------------------------------------
*/
#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <string.h>
#include <mpi.h>
#include <memory>

#include "work.h"



//Size of communication buffers in B
//-----------------------------------
//2*MAX_INTERVALS*log(2,N!)/8
//-----------------------------------
//16384*2*log(2;50!)/8 ~= 1MB
//16384*2*log(2;100!)/8 ~= 2.1MB
//16384*2*log(2;200!)/8 ~= 5.1MB
//16384*2*log(2;400!)/8 ~= 11.82MB
//16384*2*log(2;600!)/8 ~= 19.1MB
//16384*2*log(2;800!)/8 ~= 26.9MB
//-----------------------------------
//constexpr size_t MAX_COMM_BUFFER=8388608; //8MB
//constexpr size_t MAX_COMM_BUFFER=16777216; //16 MB
constexpr size_t MAX_COMM_BUFFER=33554432; //32MB
//constexpr size_t MAX_COMM_BUFFER=134217728; //128MB


//Max number of digits in factorial (log(10;N!))
constexpr size_t MAX_MPZLEN=2000; //log(10;800!) ~= 1976

//Message identifiers
constexpr int WORK=1;
constexpr int BEST=2;
constexpr int END=3;
constexpr int NIL=4;
constexpr int SLEEP=5;
constexpr int NEWWORK=6;



class communicator{
public:
    communicator(int _size): size(_size){
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    };

    int size;
    int rank;

    void send_work(std::shared_ptr<work> src_wrk, int dest, int tag);
    void recv_work(std::shared_ptr<work> dst_wrk, int src, int tag, MPI_Status* status);

    void send_sol(const int* const arr, int cost, int dest, int tag);
    void recv_sol(int* arr, int& cost, int dest, int tag, MPI_Status* status);
};

#endif
