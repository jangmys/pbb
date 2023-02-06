#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <string.h>
#include <mpi.h>
#include <memory>

#include "work.h"

class communicator{
public:
    communicator(int _M,int _size): M(_M),size(_size){
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    };

    int M; //commsize
    int size; //pbsize
    int rank; //rank

    void send_work(std::shared_ptr<work> src_wrk, int dest, int tag);
    void recv_work(std::shared_ptr<work> dst_wrk, int src, int tag, MPI_Status* status);

    void send_sol(const int* const arr, int cost, int dest, int tag);
    void recv_sol(int* arr, int& cost, int dest, int tag, MPI_Status* status);
};

#endif
