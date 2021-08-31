#ifndef INSTANCE_ABSTRACT_H
#define INSTANCE_ABSTRACT_H

#include <sstream>

struct instance_abstract {
    int                 size; //problem size
    std::stringstream   *data; //the instance data
};

#endif
