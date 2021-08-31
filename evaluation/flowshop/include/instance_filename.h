#ifndef INSTANCE_FILENAME_H
#define INSTANCE_FILENAME_H

// #include <stdlib.h>
// #include <fstream>
// #include <string.h>

struct instance_filename : public instance_abstract {
    // char * file;
    instance_filename(const char * inst_name);
    ~instance_filename();

    // int
    // get_size(const char * file);
    // void
    // generate_instance(const char * file, std::ostream& stream);
    //
    // static int
    // get_initial_ub_from_file(const char * inst_name, int init_mode);
};

#endif
