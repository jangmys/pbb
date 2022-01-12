#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <iomanip>
#include <sstream>
#include <fstream>
#include <climits>
#include <random>

#include "instance_abstract.h"
#include "instance_random.h"

//rand_20_5
//...

instance_random::instance_random(const char * inst_name)
{
    data = new std::stringstream();

    std::string s(inst_name);

    auto pos1 = s.find_first_of("_");
    auto pos2 = s.find_last_of("_");

    auto nbJob = std::stoi(s.substr(pos1+1,pos2-pos1));
    auto nbMachines = std::stoi(s.substr(pos2+1));

    size = nbJob;

    *data << nbJob << " " << nbMachines << " ";

    std::cout<<" === Nb Jobs:\t"<<nbJob<<std::endl;
    std::cout<<" === Nb Machines:\t"<<nbMachines<<std::endl;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist99(0,99);

    std::cout<<" === Processing Times:"<<std::endl;

    for(int i=0;i<nbMachines;i++){
        for(int j=0;j<nbJob;j++){
            auto p = dist99(rng);
            std::cout<<std::setw(2)<<p<<" ";

            *data<<p<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

instance_random::~instance_random()
{
    delete data;
}
