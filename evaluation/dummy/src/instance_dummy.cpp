#include <iostream>

#include <instance_abstract.h>
#include <instance_dummy.h>

instance_dummy::instance_dummy(const std::string inst_name){
    data = new std::stringstream();

    auto nbJob = std::stoi(inst_name);
    size = nbJob;

    (*data) << size;
    std::cout<<"dummy instance : "<<size<<"\n";
}

instance_dummy::~instance_dummy()
{
    std::cout<<"dummy delete\n";
    delete data;
}
