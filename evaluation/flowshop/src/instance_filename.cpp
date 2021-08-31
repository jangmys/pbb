#include <iostream>
#include <iomanip>
#include <fstream>

#include "instance_abstract.h"
#include "instance_filename.h"

instance_filename::instance_filename(const char * inst_name)
{
    data = new std::stringstream();

    std::ifstream infile(inst_name);

    int nbMachines;

    if (infile.is_open()) {
        infile.seekg(0);
        if (!infile.eof()) infile >> size;
        if (!infile.eof()) infile >> nbMachines;

        if (nbMachines) {
            *data << size << " " << nbMachines << " ";
        } else {
            perror("infile read error");
            exit(1);
        }

        int c = 0;
        do{
            infile >> c;
            *data << c << " ";
        }while(!infile.eof());

        // while (1) {
        //     infile >> c;
        //     *data << c << " ";
        //     if (infile.eof()) break;
        // }
    } else  {
        std::cout << "Error opening file: " << std::string(inst_name) << "\n";
        exit(1);
    }
}

instance_filename::~instance_filename()
{
    delete data;
}
