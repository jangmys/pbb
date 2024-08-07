#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <climits>
#include <getopt.h>


#include "log.h"
#include "../inih/INIReader.h"

class arguments
{
public:
    // B&B components
    static int branchingMode;    // --branch
    static int boundMode;// --bound

    static int primary_bound; // simple / johnson
    static int secondary_bound;

    // incumbent
    static int init_mode;
    static int initial_ub;
    static bool increaseInitialUB;

    // distributed
    static char worker_type; //cpu or gpu (distributed mode)
    static bool singleNode; //no MPI = 1 ; distributed = 0

    //heuristic
    static char heuristic_type;
    static int heuristic_threads; //number of heuristic threads
    static int heuristic_iters;
    static int initial_heuristic_iters;

    //data struct
    static char ds;

    //time
    static int checkpointv;
    static int balancingv;
    static int timeout;

    static int initial_work;
    static int sortNodes;
    static int nodePriority;
    static int nbivms_mc;//la m^me ...
    static int nbivms_gpu;//chose ...

    //problem
    static std::string problem;
    static std::string inst_name;
    static char work_directory[50];

    static char logfile[50];
    static TLogLevel logLevel;

    // problem specific (FSP) =================
    static bool earlyStopJohnson;
    static int johnsonPairs;

    static int singleNodeDS;

    static bool findAll;

    //verbosity
    static bool printSolutions;
    static int gpuverb;

    //work stealing
    static char mc_ws_select;

    static void arg_summary();

    static void readIniFile(char inifile[]);
    static void readIniFile(std::string inifile);
    static bool parse_arguments(int argc, char **argv);
};

#endif
