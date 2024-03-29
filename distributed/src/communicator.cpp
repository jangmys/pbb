/*
===================================
Send/Receive routines for work units (sets of intervals)
--------------------------------------------------------
Author : Jan Gmys (jan.gmys@univ-lille.fr)
------------------------------------------
*/
#include "macros.h"
#include "pbab.h"
#include "ttime.h"

#include "communicator.h"
#include "work.h"

//===============================================
void communicator::send_work(std::shared_ptr<work> src_wrk, int dest, int tag)
{
    char *buffer=(char*)malloc(MAX_COMM_BUFFER);
    int pos=0;

    MPI_Pack(&src_wrk->id, 1, MPI_INT, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);

    src_wrk->nb_intervals=(src_wrk->Uinterval).size();
    MPI_Pack(&src_wrk->nb_intervals, 1, MPI_INT, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);
    MPI_Pack(&src_wrk->max_intervals, 1, MPI_INT, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);
    MPI_Pack(&src_wrk->nb_decomposed, 1, MPI_INT, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);
    MPI_Pack(&src_wrk->nb_leaves, 1, MPI_INT, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);

    char ptr[MAX_MPZLEN];
    FILE* bp = fmemopen(ptr, MAX_MPZLEN, "w");

    for(auto it: src_wrk->Uinterval){
        MPI_Pack(&(it->id), 1, MPI_INT, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);

        //return number of bytes written
        size_t sz=mpz_out_raw(bp, (it->begin).get_mpz_t());
        //pack size information
        MPI_Pack(&sz, 1, MPI_INT, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);
        //pack mpz
        rewind(bp);
        MPI_Pack(ptr, sz, MPI_BYTE, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);

        //repeat for second
        sz=mpz_out_raw(bp, (it->end).get_mpz_t());
        MPI_Pack(&sz, 1, MPI_INT, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);
        rewind(bp);
        MPI_Pack(ptr, sz, MPI_BYTE, buffer, MAX_COMM_BUFFER, &pos, MPI_COMM_WORLD);
    }

    MPI_Send(buffer, pos, MPI_PACKED, dest, tag, MPI_COMM_WORLD);

    fclose(bp);
    free(buffer);
}
//===============================================
void communicator::recv_work(std::shared_ptr<work> dst_wrk, int src, int tag, MPI_Status* status){
    char *rbuffer=(char*)malloc(MAX_COMM_BUFFER);

    int ret=MPI_Recv(rbuffer, MAX_COMM_BUFFER, MPI_PACKED, src, tag, MPI_COMM_WORLD, status);
    if(ret==MPI_ERR_TRUNCATE){
        printf("buffer overflow\n");
    }

    int pos=0;
    MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, &dst_wrk->id, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, &dst_wrk->nb_intervals, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, &dst_wrk->max_intervals, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, &dst_wrk->nb_decomposed, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, &dst_wrk->nb_leaves, 1, MPI_INT, MPI_COMM_WORLD);
//    printf("received work %d, nb intervals %d\n",dst_wrk->id,dst_wrk->nb_intervals);
//    fflush(stdout);

    // size_t size=0;
    int id;

    mpz_t src_mpz;
    mpz_init(src_mpz);

    mpz_class tmpb(0);
    mpz_class tmpe(0);

    char mpz_raw[MAX_MPZLEN];
    FILE* fd1 = fmemopen(mpz_raw, MAX_MPZLEN, "r");

    size_t sz=0;

    for(int i=0;i<dst_wrk->nb_intervals;i++){
        MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, &id, 1, MPI_INT, MPI_COMM_WORLD);
        //unpack mpz size
        MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, &sz, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, mpz_raw, sz, MPI_BYTE, MPI_COMM_WORLD);

        rewind(fd1);
        mpz_inp_raw(src_mpz, fd1);
        tmpb=mpz_class(src_mpz);

        //=============
        MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, &sz, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(rbuffer, MAX_COMM_BUFFER, &pos, mpz_raw, sz, MPI_BYTE, MPI_COMM_WORLD);

        rewind(fd1);
        mpz_inp_raw(src_mpz, fd1);
        tmpe=mpz_class(src_mpz);

        (dst_wrk->Uinterval).emplace_back(std::make_shared<interval>(tmpb, tmpe, id));
    }


    mpz_clear(src_mpz);

    fclose(fd1);
    free(rbuffer);
}
//===============================================
void communicator::send_sol(const int* const arr, int cost, int dest, int tag)
{
    int *buf = new int[size+1];

    buf[0]=cost;
    memcpy(&buf[1], arr, size*sizeof(int));

    MPI_Send(buf, size+1, MPI_INT, dest, tag, MPI_COMM_WORLD);

    delete[]buf;
}
//===============================================
void communicator::recv_sol(int* arr, int& cost, int src, int tag, MPI_Status* status){
    int *buf = new int[size+1];

    MPI_Recv(buf, size+1, MPI_INT, src, tag, MPI_COMM_WORLD, status);

    cost=buf[0];
    memcpy(arr, &buf[1], size*sizeof(int));

    delete[]buf;
}
