#include <sys/stat.h>

#include "../include/log.h"
#include "../include/arguments.h"

// #include "libbounds.h"

//initialization files
char arguments::inifile[50] = "./bbconfig.ini";
char arguments::work_directory[50] = "../../bbworks/";

//instance / problem
// int arguments::instancev;
char arguments::inst_name[50];
char arguments::problem[50];

//Bounding options
int arguments::boundMode     = 2;
bool arguments::earlyStopJohnson = true;
int arguments::johnsonPairs      = 1;

//Branching options
int arguments::branchingMode = 3;
int arguments::sortNodes         = 1;
int arguments::nodePriority = 1;

//Pruning options
bool arguments::findAll        = false;

//initial upper bound and solutions
int arguments::init_solution = 1;
int arguments::init_mode = 1;
int arguments::initial_ub;

//parallel
bool arguments::singleNode = true;// false;
int arguments::nbivms_mc  = -1;
int arguments::nbivms_gpu = 4096;

//load balance / fault tolerance
int arguments::checkpointv = 1;
int arguments::balancingv  = 1;
char arguments::mc_ws_select = 'o';

//heuristic...
int arguments::heuristic_threads       = 1;
int arguments::heuristic_iters         = 1;
int arguments::initial_heuristic_iters = 100;
char arguments::heuristic_type         = 'n';

//timeout
bool arguments::mc_timeout = false;
int arguments::timeout  = 99999;

//truncation
int arguments::truncateDepth   = 0;
bool arguments::truncateSearch = false;
int arguments::cut_top         = 99999;
int arguments::cut_bottom      = -1;

//verbosity / logging
bool arguments::printSolutions = true;
char arguments::logfile[50] = "./logfile.txt";
int arguments::logLevel = logINFO;

//initial search space
int arguments::initial_work = 3;

/***********************************************************************/

std::string
sections(INIReader &reader)
{
    std::stringstream ss;
    std::set<std::string> sections = reader.Sections();
    for (std::set<std::string>::iterator it = sections.begin(); it != sections.end(); ++it)
        ss << *it << ",";
    return ss.str();
}

void
arguments::readIniFile()
{
    std::string str(inifile);

    INIReader reader(str);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load " << str << "\n";
        return;
    }
    strncpy(problem, reader.Get("problem", "problem", "UNKNOWN").c_str(), 50);
    strncpy(inst_name, reader.Get("problem", "instance", "UNKNOWN").c_str(), 50);

    checkpointv = reader.GetInteger("time", "checkpoint", 1);// default values;
    mc_timeout  = reader.GetBoolean("time", "timeout", false);// default values;
    balancingv  = reader.GetInteger("time", "balance", 1);
    timeout  = reader.GetInteger("time", "timeout", 99999);
    // if(!mc_timeout){ //timeout only for singleNode
    //   balancingv=INT_MAX;
    // }

    init_mode = reader.GetInteger("initial", "ub", -1);
    init_solution = reader.GetInteger("initial", "sol", -1);

    // nb IVMs
    // nbivm_mc   = reader.GetInteger("multicore", "threads", -1);
    nbivms_mc  = reader.GetInteger("multicore", "threads", -1);
    nbivms_gpu = reader.GetInteger("gpu", "nbIVMs", 4096);

    // sorting
    sortNodes    = reader.GetInteger("bb", "sortedDFS", 1);
    nodePriority = reader.GetInteger("bb", "sortingCriterion", 1);

    earlyStopJohnson = reader.GetBoolean("bb", "earlyStopJohnson", true);
    boundMode        = reader.GetInteger("bb", "boundingMode", 2);
    johnsonPairs     = reader.GetInteger("bb", "JohnsonMode", 1);

    findAll    = reader.GetBoolean("bb", "findAll", false);
    singleNode = reader.GetBoolean("bb", "singleNode", false);
    if (singleNode)
        std::cout << "Single-node mode" << std::endl;

    branchingMode = reader.GetInteger("bb", "adaptiveBranchingMode", 3);

    printSolutions = reader.GetBoolean("verbose", "printSolutions", false);
    // if(printSolutions)
    //     std::cout<<"Printing Solutions..."<<std::endl;

    mc_ws_select = *(reader.Get("multicore", "worksteal", "r").c_str());
    // type         = reader.Get("bb", "type", "c")[0];

    truncateDepth  = reader.GetInteger("truncate", "truncDepth", 0);
    truncateSearch = reader.GetBoolean("truncate", "truncSearch", false);
    cut_bottom     = reader.GetInteger("truncate", "cutoff_bottom", -1);
    cut_top        = reader.GetInteger("truncate", "cutoff_top", INT_MAX);

    heuristic_threads       = reader.GetInteger("heuristic", "heuristic_threads", 1);
    initial_heuristic_iters = reader.GetInteger("heuristic", "initial_heuristic_iters", 100);
    heuristic_iters         = reader.GetInteger("heuristic", "heuristic_iters", 100);
    heuristic_type = reader.Get("heuristic", "heuristic_type", "n")[0];

    initial_work = reader.GetInteger("distributed", "initialWork", 3);
} // arguments::readIniFile

inline bool
fexists(const std::string& name)
{
    struct stat buffer;

    return (stat(name.c_str(), &buffer) == 0);
}

/*read upper bounds from file*/
void
arguments::initialize()
{
    // initial_ub = INT_MAX;
    //
    // if(init_mode == 0){
    //     printf("Get initial upper bound from file\n"); fflush(stdout);
    //     switch (inst_name[0]) {
    //         case 't':
    //         {
    //             initial_ub = instance_flowshop::get_initial_ub_from_file(inst_name,init_mode);
    //             break;
    //         }
    //         case 'V':
    //         {
    //             initial_ub = instance_vrf::get_initial_ub_from_file(inst_name,init_mode);
    //             break;
    //         }
    //     }
    // }
} // arguments::initialize

#define OPTIONS "z:ftm" // vrtnqbiowcdugmsfh"
bool
arguments::parse_arguments(int argc, char ** argv)
{
    bool ok = false;

    char * subopts, * value;

    enum { PROBLEM = 0, INST, OPT };
    char * const problem_opts[] = {
        [PROBLEM] = (char *) "p",
        [INST]    = (char *) "i",
        [OPT]     = (char *) "o",
        NULL
    };

    int c = getopt_long(argc, argv, OPTIONS, NULL, NULL);

    while (c != -1) {
        switch (c) {
            case 'f': {
                strcpy(inifile, argv[optind]);
                std::cout<<"inifile:\t"<<inifile<<std::endl;
                readIniFile();
                break;
            }
            //multi-option, ex. "-z p=fsp,i=ta20,o"
            case 'z': {
                subopts = optarg;
                while (*subopts != '\0')
                    switch (getsubopt(&subopts, problem_opts, &value)) {
                        case PROBLEM:
                            strcpy(problem, value);
                            break;
                        case INST:
                            strcpy(inst_name, value);
                            break;
                        case OPT:
                            init_mode = 0;
                            break;
                    }
                ok = true;
                break;
            }
            case 't': {
                timeout = atoi(argv[optind]);
                // printf("Timeout %s\n",argv[optind]);
                // printf("Timeout %d\n",timeout);
                break;
            }
            case 'm': {
                singleNode = true;
                break;
            }
        }
        c = getopt_long(argc, argv, OPTIONS, NULL, NULL);
    }

    return ok;
} // arguments::parse_arguments
