#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <climits>

#include "instance_abstract.h"
#include "instance_vrf.h"

instance_vrf::instance_vrf(const char * inst_name)
{
    data = new std::stringstream();

    char *vrfdirname;//[30];
    vrfdirname = (char*)"../evaluation/flowshop/data/vrf_parameters/";

    // struct stat buffer;
    // if (stat(vrfdirname, &buffer) != 0){
    //     vrfdirname = (char*)"../parameters/vrf_parameters/";
    // };
    const char ext[]        = "_Gap.txt";

    file = (char *) malloc(strlen(inst_name) + strlen(vrfdirname) + strlen(ext) + 1); /* make space for the new string*/

    strcpy(file, vrfdirname);/* copy dirname into the new var */
    strcat(file, inst_name); /* add the instance name */
    strcat(file, ext);       /* add the extension */

    generate_instance(file, *data);

    free(file);
}

instance_vrf::~instance_vrf()
{
    delete data;
}

void
instance_vrf::generate_instance(const char * _file, std::ostream& stream)
{
    std::ifstream infile(_file);

    int nbMachines = 0;

    if (infile.is_open()) {
        infile.seekg(0);
        if (!infile.eof()) infile >> size;
        if (!infile.eof()) infile >> nbMachines;

        if (nbMachines) {
            stream << size << " " << nbMachines << " ";
        } else {
            perror("infile read error");
            exit(1);
        }

        int tmp[size * nbMachines];
        int c = 0;

        while (1) {
            int m;
            infile >> m >> tmp[c++];
            if (infile.eof()) break;
        }

        // transpose
        for (int i = 0; i < nbMachines; i++) {
            for (int j = 0; j < size; j++) {
                stream << tmp[j * nbMachines + i] << " ";
            }
        }
    } else  {
        std::cout << "Error opening file: " << std::string(_file) << "\n";
        exit(1);
    }
} // instance_vrf::generate_instance

int
instance_vrf::get_initial_ub_from_file(const char* inst_name,int init_mode)
{
    std::string tmp = "";
    int jobs     = 0;
    int machines = 0;
    int valeur   = 0;
    int no,lb;
    std::stringstream rubrique;
    rubrique << inst_name;// ance;

    struct stat buffer;

    std::ifstream infile;

    if (stat("../evaluation/flowshop/data/instancesVRF.data", &buffer) == 0){
        infile = std::ifstream("./evaluation/flowshop/data/instancesVRF.data");
    } else if (stat("../evaluation/flowshop/data/instancesVRF.data", &buffer) == 0)  {
        infile = std::ifstream("../evaluation/flowshop/data/instancesVRF.data");
    } else  {
        std::cout << "Trying to read best-known solution from ./parameters/instancesVRF.data failed\n";
    }

    while (!infile.eof()) {
        std::string str;
        getline(infile, str);

        if (str.substr(0, rubrique.str().length()).compare(rubrique.str()) == 0) {
            std::stringstream buffer;
            buffer << str << std::endl;
            buffer >> tmp >> jobs >> machines >> no >> valeur >> lb;
            break;
        }
    }

    infile.close();
}
