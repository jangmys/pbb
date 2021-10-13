#ifndef INSTANCE_VFR_H
#define INSTANCE_VFR_H

#include <stdlib.h>
#include <fstream>
#include <string.h>

struct instance_vrf : public instance_abstract {
    char * file;

    instance_vrf(const char * inst_name);
    ~instance_vrf();

    int
    get_size(const char * file);
    void
    generate_instance(const char * file, std::ostream& stream);

    int
    get_initial_ub_from_file(const char * inst_name, int init_mode);
};

#endif // ifndef INSTANCE_VFR_H
