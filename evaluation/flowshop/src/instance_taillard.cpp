/*
 * Flowshop instances defined by Taillard'93
 */

// #include "instance_abstract.h"
#include "instance_taillard.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <climits>
#include <sys/stat.h>

long time_seeds[] =
{ 873654221 /*ta001*/,    379008056 /*ta002*/,  1866992158 /*ta003*/, 216771124 /*ta004*/,  495070989 /*ta005*/,
    402959317 /*ta006*/,  1369363414 /*ta007*/, 2021925980 /*ta008*/, 573109518 /*ta009*/,  88325120 /*ta010*/,
    587595453 /*ta011*/,  1401007982 /*ta012*/, 873136276 /*ta013*/,  268827376 /*ta014*/,  1634173168 /*ta015*/,
    691823909 /*ta016*/,  73807235 /*ta017*/,   1273398721 /*ta018*/, 2065119309 /*ta019*/, 1672900551 /*ta020*/,
    479340445 /*ta021*/,  268827376 /*ta022*/,  1958948863 /*ta023*/, 918272953 /*ta024*/,  555010963 /*ta025*/,
    2010851491 /*ta026*/, 1519833303 /*ta027*/, 1748670931 /*ta028*/, 1923497586 /*ta029*/, 1829909967 /*ta030*/,
    1328042058 /*ta031*/, 200382020 /*ta032*/,  496319842 /*ta033*/,  1203030903 /*ta034*/, 1730708564 /*ta035*/,
    450926852 /*ta036*/,  1303135678 /*ta037*/, 1273398721 /*ta038*/, 587288402 /*ta039*/,  248421594 /*ta040*/,
    1958948863 /*ta041*/, 575633267 /*ta042*/,  655816003 /*ta043*/,  1977864101 /*ta044*/, 93805469 /*ta045*/,
    1803345551 /*ta046*/, 49612559 /*ta047*/,   1899802599 /*ta048*/, 2013025619 /*ta049*/, 578962478 /*ta050*/,
    1539989115 /*ta051*/, 691823909 /*ta052*/,  655816003 /*ta053*/,  1315102446 /*ta054*/, 1949668355 /*ta055*/,
    1923497586 /*ta056*/, 1805594913 /*ta057*/, 1861070898 /*ta058*/, 715643788 /*ta059*/,  464843328 /*ta060*/,
    896678084 /*ta061*/,  1179439976 /*ta062*/, 1122278347 /*ta063*/, 416756875 /*ta064*/,  267829958 /*ta065*/,
    1835213917 /*ta066*/, 1328833962 /*ta067*/, 1418570761 /*ta068*/, 161033112 /*ta069*/,  304212574 /*ta070*/,
    1539989115 /*ta071*/, 655816003 /*ta072*/,  960914243 /*ta073*/,  1915696806 /*ta074*/, 2013025619 /*ta075*/,
    1168140026 /*ta076*/, 1923497586 /*ta077*/, 167698528 /*ta078*/,  1528387973 /*ta079*/, 993794175 /*ta080*/,
    450926852 /*ta081*/,  1462772409 /*ta082*/, 1021685265 /*ta083*/, 83696007 /*ta084*/,   508154254 /*ta085*/,
    1861070898 /*ta086*/, 26482542 /*ta087*/,   444956424 /*ta088*/,  2115448041 /*ta089*/, 118254244 /*ta090*/,
    471503978 /*ta091*/,  1215892992 /*ta092*/, 135346136 /*ta093*/,  1602504050 /*ta094*/, 160037322 /*ta095*/,
    551454346 /*ta096*/,  519485142 /*ta097*/,  383947510 /*ta098*/,  1968171878 /*ta099*/, 540872513 /*ta100*/,
    2013025619 /*ta101*/, 475051709 /*ta102*/,  914834335 /*ta103*/,  810642687 /*ta104*/,  1019331795 /*ta105*/,
    2056065863 /*ta106*/, 1342855162 /*ta107*/, 1325809384 /*ta108*/, 1988803007 /*ta109*/, 765656702 /*ta110*/,
    1368624604 /*ta111*/, 450181436 /*ta112*/,  1927888393 /*ta113*/, 1759567256 /*ta114*/, 606425239 /*ta115*/,
    19268348 /*ta116*/,   1298201670 /*ta117*/, 2041736264 /*ta118*/, 379756761 /*ta119*/,  28837162 /*ta120*/ };


instance_taillard::instance_taillard(const char * inst)
{
    // expect instance-name to be "taX", where X is the number of Ta instance...
    int id = atoi(&inst[2]);

    data = new std::stringstream();
    size = get_job_number(id);
    generate_instance(id, *data);
}

int
instance_taillard::get_job_number(int id)
{
    if (id > 110) return 500;

    if (id > 90) return 200;

    if (id > 60) return 100;

    if (id > 30) return 50;

    /*if(id>0)*/ return 20;
}

int
instance_taillard::get_machine_number(int id)
{
    if (id > 110) return 20;

    if (id > 100) return 20;

    if (id > 90) return 10;

    if (id > 80) return 20;

    if (id > 70) return 10;

    if (id > 60) return 5;

    if (id > 50) return 20;

    if (id > 40) return 10;

    if (id > 30) return 5;

    if (id > 20) return 20;

    if (id > 10) return 10;

    /*if(id>0 )*/ return 5;
}

long
instance_taillard::unif(long * seed, long low, long high)
{
    long m = 2147483647, a = 16807, b = 127773, c = 2836, k;
    double value_0_1;

    k       = (*seed) / b;
    *(seed) = a * (*(seed) % b) - k * c;
    if ((*seed) < 0) *(seed) = *(seed) + m;
    value_0_1 = float(*seed) / float(m);
    return low + long(value_0_1 * (high - low + 1));
}

void
instance_taillard::generate_instance(int id, std::ostream& stream)
{
    long N         = get_job_number(id);
    long M         = get_machine_number(id);
    long time_seed = time_seeds[id - 1];

    // stream "data" contains: #JOBS #MACHINES   PROCESSING TIME MATRIX (M-N order)
    stream << N << " " << M << " ";
    for (int j = 0; j < M; j++)
        for (int i = 0; i < N; i++)
            stream << unif(&time_seed, 1, 99) << " ";
}

inline bool
f_exists(const std::string& name)
{
    struct stat buffer;

    return (stat(name.c_str(), &buffer) == 0);
}


int
instance_taillard::get_initial_ub_from_file(const char* inst_name, int init_mode)
{
    std::stringstream rubrique;
    std::string tmp;
    int jobs, machines, valeur, no, lb;
    std::ifstream infile;
    int initial_ub = INT_MAX;

    if (f_exists("../../evaluation/flowshop/data/instances.data")) {
        infile = std::ifstream("../../evaluation/flowshop/data/instances.data");
    } else  {
        std::cout << "Trying to read best-known solution from ../../evaluation/flowshop/data/instances.data failed\n";
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
}
