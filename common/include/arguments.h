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
    // B&B components (all B&B's)
    static int branchingMode;  // --branch
    static int boundMode;// --bound

    static int primary_bound; // --primary-bound simple 0 / johnson 1
    static int secondary_bound; // only = 1 (johnson)

    static bool findAll; // prune on equality?


    // incumbent o=
    static int init_mode;  //-z o= suboption
    static int initial_ub;
    static bool increaseInitialUB;

    //problem
    static std::string problem;
    static std::string inst_name;

    //work stealing
    static char mc_ws_select; //work stealing selection strategy (default : 'a' (random))

    //verbosity & logging
    static bool printSolutions;
    static int gpuverb;
    static char logfile[50];
    static TLogLevel logLevel;

    // dbb : distributed only
    static char worker_type; //cpu or gpu (distributed mode)
    static bool singleNode; //no MPI = 1 ; distributed = 0

    static int initial_work; // how to split initial work interval
    static char work_directory[50]; //work units read from file

    //dbb : time (checkpointing )
    static int checkpointv;
    static int balancingv;
    static int timeout;





    //heuristic (in parallel to dbb)
    static char heuristic_type;
    static int heuristic_threads; //number of heuristic threads
    static int heuristic_iters;
    static int initial_heuristic_iters;

    //data struct
    static char ds;


    static int sortNodes; //=0
    static int nodePriority;
    static int nbivms_mc;//la m^me ...
#ifdef WITH_GPU
    static int nbivms_gpu;//chose ...
#endif


    // problem specific (FSP - Johnson bound) =================
    static bool earlyStopJohnson;
    static int johnsonPairs;

    static void arg_summary();

    static void readIniFile(char inifile[]);
    static void readIniFile(std::string inifile);
    static bool parse_arguments(int argc, char **argv);
};

#endif
