#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <string.h>
#include <mpi.h>
#include <memory>


class work;
class solution;

class communicator{
public:
    communicator(int _M,int _size): M(_M),size(_size){
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    };

    int M;
    int size;
    int rank;

    void send_fwork(int dest, int tag);
    void recv_fwork(int src, int tag, MPI_Status* status);

    void send_work(std::shared_ptr<work> src_wrk, int dest, int tag);
    void recv_work(std::shared_ptr<work> dst_wrk, int src, int tag, MPI_Status* status);

    void send_sol(solution* sol, int dest, int tag);
    void recv_sol(solution* sol, int dest, int tag, MPI_Status* status);
};

#endif
