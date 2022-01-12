//
// Taillard's flowshop instances
//
#ifndef INSTANCE_TAILLARD_H
#define INSTANCE_TAILLARD_H

#include "instance_abstract.h"

#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <climits>

inline bool
f_exists(const std::string& name)
{
    struct stat buffer;

    return (stat(name.c_str(), &buffer) == 0);
}

struct instance_taillard : public instance_abstract {
    instance_taillard(const char * inst);

    int get_job_number(int id);
    int get_machine_number(int id);

    long
    unif(long * seed, long low, long high);

    void
    generate_instance(int id, std::ostream& stream);

    int read_initial_ub_from_file(const char * inst_name)
    {
        std::stringstream rubrique;
        std::string tmp;
        int jobs, machines, valeur;
        std::ifstream infile;
        int initial_ub = INT_MAX;

        if (f_exists("../evaluation/flowshop/data/instances.data")) {
            infile = std::ifstream("../evaluation/flowshop/data/instances.data");
        } else  {
            std::cout << "Trying to read best-known solution from ../evaluation/flowshop/data/instances.data failed\n";
        }

        int id = atoi(&inst_name[2]);
        rubrique << "instance" << id << "i";

        while (!infile.eof()) {
            std::string str;
            getline(infile, str);

            if (str.substr(0, rubrique.str().length()).compare(rubrique.str()) == 0) {
                std::stringstream buffer;
                buffer << str << std::endl;
                buffer >> tmp >> jobs >> machines >> valeur;
                break;
            }
        }
        infile.close();
        initial_ub = valeur;

        return initial_ub;
    };
};

#endif // ifndef INSTANCE_FLOWSHOP_H
