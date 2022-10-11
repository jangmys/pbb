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
    //heuristic
    static int heuristic_threads;
    static int heuristic_iters;
    static int initial_heuristic_iters;
    static char heuristic_type;

    // --branch
    static int branchingMode;
    // --bound
    static int boundMode;
    //
    static int primary_bound;
    static int secondary_bound;

    // static char type;       //'g'PU or 'c'PU
    static bool singleNode; //no MPI = 1 ; distributed = 0

    static int checkpointv;
    static int balancingv;

    static bool mc_timeout;
    static int timeout;


    //UB initialization
    static int init_mode;
    static int initial_ub;

    static int initial_work;
    static int sortNodes;
    static int nodePriority;
    static int nbivms_mc;//la m^me ...
    static int nbivms_gpu;//chose ...

    //problem
    static char problem[50];
    static char inst_name[50];
    static char inifile[50];
    static char work_directory[50];

    static char logfile[50];
    static int logLevel;

    //FSP - johnson bound
    static bool earlyStopJohnson;
    static int johnsonPairs;

    static int singleNodeDS;

    static bool findAll;

    //verbosity
    static bool printSolutions;

    //truncate...
    static bool truncateSearch;
    static int truncateDepth;
    static int cut_bottom;
    static int cut_top;

    //work stealing
    static char mc_ws_select;

    static void readIniFile();
    static bool parse_arguments(int argc, char **argv);
};

#endif
