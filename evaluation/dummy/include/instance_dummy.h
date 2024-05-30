#ifndef INSTANCE_DUMMY_H
#define INSTANCE_DUMMY_H

struct instance_dummy : public instance_abstract {
    instance_dummy(const std::string inst_name);
    ~instance_dummy();
};

#endif
