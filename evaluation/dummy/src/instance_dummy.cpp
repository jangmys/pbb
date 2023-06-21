#include <iostream>

#include <instance_abstract.h>
#include <instance_dummy.h>

instance_dummy::instance_dummy(const char * inst_name){
    data = new std::stringstream();

    std::string s(inst_name);

    auto nbJob = std::stoi(s);
    size = nbJob;

    (*data) << size;

    // std::cout<<"dummy instance : "<<nbJob<<" "<<size<<"\n";
}

instance_dummy::~instance_dummy()
{
    std::cout<<"dummy delete\n";
    delete data;
}
