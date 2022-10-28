/*
 * Flowshop instances defined by Taillard'93
 */
#include "c_taillard.h"

#include "instance_abstract.h"
#include "instance_taillard.h"

#include <pthread.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <climits>
#include <sys/stat.h>

instance_taillard::instance_taillard(const char * inst)
{
    // expect instance-name to be "taX", where X is the number of Ta instance...
    int id = atoi(&inst[2]);

    data = new std::stringstream();
    size = get_job_number(id);
    generate_instance(id, *data);
}

int
instance_taillard::get_job_number(int id)
{
    return taillard_get_nb_jobs(id);
}

int
instance_taillard::get_machine_number(int id)
{
    return taillard_get_nb_machines(id);
}

void
instance_taillard::generate_instance(int id, std::ostream& stream)
{
    long N         = (long)taillard_get_nb_jobs(id);//get_job_number(id);
    long M         = (long)taillard_get_nb_machines(id);
    long time_seed = time_seeds[id - 1];

    // stream "data" contains: #JOBS #MACHINES   PROCESSING TIME MATRIX (M-N order)
    stream << N << " " << M << " ";
    for (int j = 0; j < M; j++)
        for (int i = 0; i < N; i++)
            stream << unif(&time_seed, 1, 99) << " ";
}
