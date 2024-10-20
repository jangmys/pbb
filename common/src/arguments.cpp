#include <sys/stat.h>
#include <unistd.h>

#include "../include/log.h"
#include "../include/arguments.h"

//initialization files
char arguments::work_directory[50] = "../../bbworks/";

//instance / problem
std::string arguments::inst_name="ta20";
std::string arguments::problem="flowshop";

char arguments::worker_type='c'; // default : CPU

//Bounding options
int arguments::boundMode     = 0;
int arguments::primary_bound     = 0;
int arguments::secondary_bound   = 1;

bool arguments::earlyStopJohnson = true;
int arguments::johnsonPairs      = 0;

//Branching options
int arguments::branchingMode = 1;
int arguments::sortNodes         = -1;

//Pruning options
bool arguments::findAll        = false;

//Data struct
char arguments::ds = 'i';

//initial upper bound and solutions
int arguments::init_mode = 1;
int arguments::initial_ub;
bool arguments::increaseInitialUB = false;

//parallel
bool arguments::singleNode = true;// false;
int arguments::nbivms_mc  = -1;

#ifdef WITH_GPU
int arguments::nbivms_gpu = 16384;
#endif

//load balance / fault tolerance
int arguments::checkpointv = 3600;
int arguments::balancingv  = 1;
int arguments::timeout  = 99999;

char arguments::mc_ws_select = 'a'; //random

//heuristic...
int arguments::heuristic_threads       = 0;
int arguments::heuristic_iters         = 1;
int arguments::initial_heuristic_iters = 100;
char arguments::heuristic_type         = 'n';

//-------------------------------timeout-------------------------------

//------------------------- verbosity / logging-------------------------
//write new solutions to stdout
bool arguments::printSolutions = false;
//logfile
char arguments::logfile[50] = "./logfile.txt";
//logging level : error < info < debug < debug4
TLogLevel arguments::logLevel = logINFO;
//GPU execution output every N iterations (0 : off)
int arguments::gpuverb=0;

//initial search space (distributed)
int arguments::initial_work = 3;

/***********************************************************************/

void read_init_mode(char* init_mode_str, int& init_mode, int& initial_ub)
{
    initial_ub = INT_MAX;
    if(init_mode_str){
        if(*init_mode_str == 'f'){
            init_mode = 0; //read opt from file
        }
        else if(*init_mode_str == 'i'){
            init_mode = -1; //infinity
        }
        else if(strcmp(init_mode_str,"neh") == 0){
            init_mode = 1; // NEH
        }else if(strcmp(init_mode_str,"beam") == 0){
            init_mode = 2; //BEAM search
        }else{
            init_mode = -1; //set value
            initial_ub = atoi(init_mode_str);
        }
    }
}

bool file_exists (char *filename) {
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

void
arguments::readIniFile(char inifile[])
{
    if(file_exists(inifile)){
        printf("\tReading config file %s.\n",inifile);
        printf("\tCommand-line options overwrite options from config file!\n");

        std::string file(inifile);
        readIniFile(file);
    }else{
        printf("File %s not found\n",inifile);
    }
}

void
arguments::readIniFile(std::string inifile)
{
    INIReader reader(inifile);
    if (reader.ParseError() < 0) {
        std::cout << "Something wrong opening " << inifile << "\n";
        return;
    }

    // ---------------------------problem definition---------------------------
    problem = reader.Get("problem", "problem", "UNKNOWN");
    inst_name = reader.Get("problem", "instance", "UNKNOWN");

    char init_mode_str[50];
    strncpy(init_mode_str, reader.Get("problem", "init_ub", "UNKNOWN").c_str(), 49);
    read_init_mode(init_mode_str, init_mode, initial_ub);

    // -----------------checkpoint / load balancing intervals -----------------
    checkpointv = reader.GetInteger("time", "checkpoint", checkpointv);// use default values;
    balancingv  = reader.GetInteger("time", "balance", balancingv);
    timeout  = reader.GetInteger("time", "timeout", timeout);

    // ------------------------nb concurrent explorers------------------------
    nbivms_mc  = reader.GetInteger("multicore", "threads", nbivms_mc);
#ifdef WITH_GPU
    nbivms_gpu = reader.GetInteger("gpu", "nbIVMs", nbivms_gpu);
#endif
    // ---------------------------sort sibling nodes---------------------------
    sortNodes    = reader.GetInteger("bb", "sortedDFS", sortNodes);

    // ----------------------------------bound----------------------------------
    primary_bound = reader.GetInteger("bb", "primaryBound", primary_bound);
    secondary_bound = reader.GetInteger("bb", "secondaryBound", secondary_bound);

    // -----------------------------johnson bound-----------------------------
    johnsonPairs     = reader.GetInteger("bb", "JohnsonMode", johnsonPairs);
    earlyStopJohnson = reader.GetBoolean("bb", "earlyStopJohnson", earlyStopJohnson);
    boundMode        = reader.GetInteger("bb", "boundingMode", boundMode);

    //
    findAll    = reader.GetBoolean("bb", "findAll", findAll);

    // single
    singleNode = reader.GetBoolean("bb", "singleNode", singleNode);

    // --------------------------------branching--------------------------------
    branchingMode = reader.GetInteger("bb", "adaptiveBranchingMode", branchingMode);

    // -------------------------------verbosity---------------------------------
    printSolutions = reader.GetBoolean("verbose", "printSolutions", printSolutions);
    // if(printSolutions)
    //     std::cout<<"Printing Solutions..."<<std::endl;

    mc_ws_select = *(reader.Get("multicore", "worksteal", "a").c_str());
    // type         = reader.Get("bb", "type", "c")[0];

    // --------------------------------heuristic--------------------------------
    heuristic_threads       = reader.GetInteger("heuristic", "heuristic_threads", heuristic_threads);
    initial_heuristic_iters = reader.GetInteger("heuristic", "initial_heuristic_iters", initial_heuristic_iters);
    heuristic_iters         = reader.GetInteger("heuristic", "heuristic_iters", heuristic_iters);
    heuristic_type = reader.Get("heuristic", "heuristic_type", "n")[0];

    initial_work = reader.GetInteger("distributed", "initialWork", initial_work);
}


#define OPTIONS "z:t:mabf:" // vrtnqbiowcdugmsfh"
bool
arguments::parse_arguments(int argc, char ** argv)
{


    bool ok = false;

    enum { PROBLEM = 0, INST, OPT };
    char * const problem_opts[] = {
        [PROBLEM] = (char *) "p",
        [INST]    = (char *) "i",
        [OPT]     = (char *) "o",
        NULL
    };

    static struct option long_options[] = {
                {"bound",   required_argument, NULL,  0 },
                {"branch",  required_argument, NULL,  0 },
                {"findall",  no_argument, NULL,  0 },
                {"print-sol",  no_argument, NULL,  0 },
                {"singlenode",  no_argument, NULL,  0 },
                {"primary-bound",  required_argument, NULL,  0 },
                {"gpu", optional_argument, NULL, 0},
                {"ll", no_argument, NULL, 0},
                {"inc-initial-ub", no_argument, NULL, 0},
                {"file",required_argument,NULL, 0},
                {"heuristic-threads",  required_argument, NULL,  0 },
                {"sortDFS",  required_argument, NULL,  0 },
                {0,         0,                 0,  0 }
            };

    int option_index;
    int c;

    while ( (c=getopt_long(argc, argv, OPTIONS, long_options, &option_index)) != -1) {
        switch (c) {
        case 0: //it's a long_option
        {
            if(strcmp(long_options[option_index].name,"file") == 0)
            {
                if(optind == 3){
                    readIniFile(optarg);
                }else{
                    printf("Aborting : config file must be given as first option.\n");
                    exit(-1);
                }
            }
            // --gpu=<nbivm_gpu>
            if(strcmp(long_options[option_index].name,"gpu") == 0)
            {
#ifdef WITH_GPU
                worker_type='g';
                //how many GPU workers ?
                nbivms_gpu=(optarg == NULL) ? 4096 : atoi(optarg);
#else
                printf("Not compiled with -DGPU. Cannot set worker type to GPU.\n");
#endif

            }
            if(strcmp(long_options[option_index].name,"ll") == 0)
            {
                ds='p';
            }
            if(strcmp(long_options[option_index].name,"bound") == 0)
            {
                boundMode = atoi(optarg);
                if(boundMode==2){
                    primary_bound = 0;
                    secondary_bound = 1;
                }
            }
            else if(strcmp(long_options[option_index].name,"branch")  == 0)
            {
                branchingMode = atoi(optarg);
            }
            else if(strcmp(long_options[option_index].name,"findall")  == 0)
            {
                findAll = true;
            }
            else if(strcmp(long_options[option_index].name,"print-sol")  == 0)
            {
                printSolutions = true;
            }
            else if(strcmp(long_options[option_index].name,"singlenode")  == 0)
            {
                singleNode = true;
            }
            else if(strcmp(long_options[option_index].name,"inc-initial-ub")  == 0)
            {
                increaseInitialUB = true;
            }
            else if(strcmp(long_options[option_index].name,"sortDFS")  == 0)
            {
                sortNodes = atoi(optarg);
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
            else if(strcmp(long_options[option_index].name,"heuristic-threads")  == 0)
            {
                heuristic_threads = atoi(optarg);
            }
            break;
        }
        //multi-option, ex. "-z p=fsp,i=ta20,o"
        case 'z': {
            char * subopts, * value;

            subopts = optarg;
            while (*subopts != '\0'){
                switch (getsubopt(&subopts, problem_opts, &value)) {
                case PROBLEM:
                    problem = std::string(value);
                    break;
                case INST:
                    inst_name = std::string(value);
                    break;
                case OPT:
                    read_init_mode(value,init_mode,initial_ub);
                    break;
                }
            }
            ok = true;
            break;
        }
        case 't': {
            // std::cout<<"option -t "<<optind<<" "<<atoi(argv[optind])<<" "<<optarg<<" "<<atoi(optarg)<<"\n";
            // printf("option t %d\n",  atoi(argv[optind]));
            nbivms_mc = atoi(optarg);
            break;
        }
        case 'm': { //already in longopt
            singleNode = true;
            break;
        }
        case 'a': { //already in longopt
            findAll = true;
            break;
        }
        case 'f':
        {
            if(optind == 3){
                readIniFile(optarg);
            }else{
                printf("Aborting : config file must be given as first option.\n");
                exit(-1);
            }
            break;
        }
        }
        // c = getopt_long(argc, argv, OPTIONS, long_options, &option_index);
    }

    return ok;
}

void arguments::arg_summary()
{
    if (singleNode){
        std::cout << "Single-node mode" << std::endl;
    }

    // stdout
    std::cout<<"Problem:\t\t"<<arguments::problem<<" / Instance "<<arguments::inst_name<<"\n";
    std::cout<<"Worker type:\t\t"<<arguments::worker_type<<std::endl;
    if(arguments::worker_type=='g'){
#ifdef WITH_GPU
        std::cout<<"#GPU workers:\t\t"<<arguments::nbivms_gpu<<std::endl;
#else
        std::cout<<"Not compiled with -DGPU but arguments::worker_type=='g'"<<std::endl;
#endif
    }
    else if(arguments::worker_type=='c'){
        std::cout<<"#CPU threads:\t\t"<<arguments::nbivms_mc<<std::endl;
    }

    std::cout<<"Bounding mode:\t\t"<<arguments::boundMode<<std::endl;
    //if Johnson bound is Used (PFSP)
    if(arguments::primary_bound == 1 || (arguments::boundMode == 2 && arguments::secondary_bound == 1))
    {
        std::cout<<"\t#Johnson Pairs:\t\t"<<arguments::johnsonPairs<<std::endl;
        std::cout<<"\tEarly Exit:\t\t"<<arguments::earlyStopJohnson<<std::endl;
    }
    std::cout<<"Branching:\t\t"<<arguments::branchingMode<<std::endl;

    //===============================================================================================
    if (singleNode){
        FILE_LOG(logINFO) << "Single-node mode" << std::endl;
    }

    // stdout
    FILE_LOG(logINFO)<<"Problem:\t\t"<<arguments::problem<<" / Instance "<<arguments::inst_name;
    FILE_LOG(logINFO)<<"Worker type:\t\t"<<arguments::worker_type;
    if(arguments::worker_type=='g'){
#ifdef WITH_GPU
        FILE_LOG(logINFO)<<"#GPU workers:\t\t"<<arguments::nbivms_gpu;
#else
        std::cout<<"Not compiled with -DGPU but arguments::worker_type=='g'"<<std::endl;
#endif
    }
    else if(arguments::worker_type=='c'){
        FILE_LOG(logINFO)<<"#CPU threads:\t\t"<<arguments::nbivms_mc;
    }

    FILE_LOG(logINFO)<<"Bounding mode:\t\t"<<arguments::boundMode;
    if(arguments::primary_bound == 1 || (arguments::boundMode == 2 && arguments::secondary_bound == 1))
    {
        FILE_LOG(logINFO)<<"\t#Johnson Pairs:\t\t"<<arguments::johnsonPairs;
        FILE_LOG(logINFO)<<"\tEarly Exit:\t\t"<<arguments::earlyStopJohnson;
    }
    FILE_LOG(logINFO)<<"Branching:\t\t"<<arguments::branchingMode;
}
