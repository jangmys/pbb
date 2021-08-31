//
// Taillard's flowshop instances
//
#ifndef INSTANCE_FLOWSHOP_H
#define INSTANCE_FLOWSHOP_H

#include "instance_abstract.h"

struct instance_taillard : public instance_abstract {
    instance_taillard(const char * inst);

    int get_job_number(int id);
    int get_machine_number(int id);

    long
    unif(long * seed, long low, long high);

    void
    generate_instance(int id, std::ostream& stream);

    static int
    get_initial_ub_from_file(const char * inst_name, int init_mode);
};

#endif // ifndef INSTANCE_FLOWSHOP_H
