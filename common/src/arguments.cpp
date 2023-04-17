#include <sys/stat.h>

#include "../include/log.h"
#include "../include/arguments.h"

//initialization files
char arguments::inifile[50] = "../bbconfig.ini";
char arguments::work_directory[50] = "../../bbworks/";

//instance / problem
char arguments::inst_name[50];
char arguments::problem[50];

char arguments::worker_type='c';

//Bounding options
int arguments::boundMode     = 2;
int arguments::primary_bound     = 0;
int arguments::secondary_bound   = 1;

bool arguments::earlyStopJohnson = true;
int arguments::johnsonPairs      = 0;

//Branching options
int arguments::branchingMode = 3;
int arguments::sortNodes         = 1;
int arguments::nodePriority = 1;

//Pruning options
bool arguments::findAll        = false;

//Data struct
char arguments::ds = 'i';

//initial upper bound and solutions
int arguments::init_mode = 1;
int arguments::initial_ub;

//parallel
bool arguments::singleNode = true;// false;
int arguments::nbivms_mc  = -1;
int arguments::nbivms_gpu = 4096;

//load balance / fault tolerance
int arguments::checkpointv = 1;
int arguments::balancingv  = 1;
char arguments::mc_ws_select = 'a'; //random

//heuristic...
int arguments::heuristic_threads       = 1;
int arguments::heuristic_iters         = 1;
int arguments::initial_heuristic_iters = 100;
char arguments::heuristic_type         = 'n';

//timeout
bool arguments::mc_timeout = false;
int arguments::timeout  = 99999;

//verbosity / logging
bool arguments::printSolutions = false;
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

    nbivms_mc  = reader.GetInteger("multicore", "threads", -1);
    nbivms_gpu = reader.GetInteger("gpu", "nbIVMs", 4096);

    // sorting
    sortNodes    = reader.GetInteger("bb", "sortedDFS", 1);
    nodePriority = reader.GetInteger("bb", "sortingCriterion", 1);

    //johnson bound
    earlyStopJohnson = reader.GetBoolean("bb", "earlyStopJohnson", true);
    boundMode        = reader.GetInteger("bb", "boundingMode", 2);
    primary_bound = reader.GetInteger("bb", "primaryBound", 0);
    secondary_bound = reader.GetInteger("bb", "secondaryBound", 1);

    johnsonPairs     = reader.GetInteger("bb", "JohnsonMode", 1);

    findAll    = reader.GetBoolean("bb", "findAll", false);
    singleNode = reader.GetBoolean("bb", "singleNode", false);
    if (singleNode)
        std::cout << "Single-node mode" << std::endl;

    branchingMode = reader.GetInteger("bb", "adaptiveBranchingMode", 3);

    printSolutions = reader.GetBoolean("verbose", "printSolutions", false);
    // if(printSolutions)
    //     std::cout<<"Printing Solutions..."<<std::endl;

    mc_ws_select = *(reader.Get("multicore", "worksteal", "a").c_str());
    // type         = reader.Get("bb", "type", "c")[0];


    heuristic_threads       = reader.GetInteger("heuristic", "heuristic_threads", 1);
    initial_heuristic_iters = reader.GetInteger("heuristic", "initial_heuristic_iters", 100);
    heuristic_iters         = reader.GetInteger("heuristic", "heuristic_iters", 100);
    heuristic_type = reader.Get("heuristic", "heuristic_type", "n")[0];

    initial_work = reader.GetInteger("distributed", "initialWork", 3);
} // arguments::readIniFile

// inline bool
// fexists(const std::string& name)
// {
//     struct stat buffer;
//
//     return (stat(name.c_str(), &buffer) == 0);
// }


#define OPTIONS "z:ftmab" // vrtnqbiowcdugmsfh"
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

    int option_index=0;
    static struct option long_options[] = {
                {"bound",   required_argument, NULL,  0 },
                {"branch",  required_argument, NULL,  0 },
                {"findall",  no_argument, NULL,  0 },
                {"singlenode",  no_argument, NULL,  0 },
                {"primary-bound",  required_argument, NULL,  0 },
                {"gpu", no_argument, NULL, 0},
                {"ll", no_argument, NULL, 0},
                {0,         0,                 0,  0 }
            };

    int c = getopt_long(argc, argv, OPTIONS, long_options, &option_index);

    while (c != -1) {
        switch (c) {
            case 0: //long_options
            {
                if(strcmp(long_options[option_index].name,"gpu") == 0)
                {
                    worker_type='g';
                }
                if(strcmp(long_options[option_index].name,"ll") == 0)
                {
                    ds='p';
                }
                if(strcmp(long_options[option_index].name,"bound") == 0)
                {
                    boundMode = atoi(optarg);
                }
                else if(strcmp(long_options[option_index].name,"branch")  == 0)
                {
                    branchingMode = atoi(optarg);
                }
                else if(strcmp(long_options[option_index].name,"findall")  == 0)
                {
                    findAll = true;
                }
                else if(strcmp(long_options[option_index].name,"singlenode")  == 0)
                {
                    singleNode = true;
                }
                else if(strcmp(long_options[option_index].name,"primary-bound") == 0)
                {
                    if(optarg[0]=='j')
                    {
                        primary_bound = 1;
                    }else{
                        primary_bound = 0;
                    }
                    // include johnson option in format "j:0:1:1"
                    // printf(" == primary-bound %c\n",optarg[0]);
                    // printf(" == primary-bound %c\n",optarg[2]);
                    // printf(" == primary-bound %c\n",optarg[4]);
                }

                break;
            }

            //-f ../multicore/mcconfig.ini
            case 'f': {
                strcpy(inifile, argv[optind]);
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
                            if(value){
                                if(*value == 'f'){
                                    init_mode = 0;
                                    printf("file\n");
                                }
                                else if(*value == 'i'){
                                    init_mode = -1;
                                    initial_ub = INT_MAX;
                                    printf("infty\n");
                                }
                                else if(strcmp(value,"neh") == 0){
                                    init_mode = 1;
                                }else if(strcmp(value,"beam") == 0){
                                    init_mode = 2;
                                }else{
                                    printf("value\n");
                                    init_mode = -1;
                                    initial_ub = atoi(value);
                                }
                            }
                            printf("%d\n",initial_ub);

                            break;
                    }
                ok = true;
                break;
            }
            case 't': {
                nbivms_mc = atoi(argv[optind]);
                break;
            }
            case 'm': {
                singleNode = true;
                break;
            }
            case 'a': {
                findAll = true;
                break;
            }
        }
        c = getopt_long(argc, argv, OPTIONS, long_options, &option_index);
    }

    return ok;
} // arguments::parse_arguments
