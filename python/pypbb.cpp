#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>

#include "libbounds.h"
#include "subproblem.h"
#include "rand.hpp"
#include "make_ivm_algo.h"

#include "libheuristic.h"
#include "matrix_controller.h"

namespace py = pybind11;


class test_abstract {
public:
    virtual ~test_abstract() = default;

    virtual void init(instance_abstract& a) = 0;

    virtual int foo() = 0;
};

class test : public test_abstract {
public:
    ~test(){};

    test(){};

    void init(instance_abstract& a){
        std::cout<<"test";
    }

    int foo(){return 42;};

private:
    int* ptr;

};


void print_vector(std::vector<int> v){
    for(auto &&a : v){
        std::cout<<a<<" ";
    }
    std::cout<<std::endl;
}


#include "log.h"

PYBIND11_MODULE(pypbb, m) {
    FILELog::ReportingLevel() = logINFO;

    m.doc() = "pybind11 example plugin"; // optional module docstring

    // //=============================================================
    py::class_<arguments>(m, "args")
        .def(py::init<>())
        // .def_static("parse",&arguments::parse_arguments)
        .def_readwrite_static("branch",&arguments::branchingMode)
        .def_readwrite_static("bound",&arguments::boundMode)
        .def_readwrite_static("threads",&arguments::nbivms_mc)
        .def_readwrite_static("problem",&arguments::problem)
        .def_readwrite_static("inst_name",&arguments::inst_name)
        .def_readwrite_static("ws",&arguments::mc_ws_select)
        ;


    //expose subproblem
    //>>  s=module.subproblem(20,[1,3,5,7,9,11,13,15,17,19,0,2,4,6,8,10,12,14,16,18])
    py::class_<subproblem,std::shared_ptr<subproblem>>(m, "subproblem")
        .def(py::init<int>())
        .def(py::init<const int, const std::vector<int>>())
        .def("__str__",
                [](const subproblem& a) {
                    std::ostringstream stream;
                    stream << a;
                    return stream.str();
                    // return "<example.Pet named '" + a.name + "'>";
                }
            )
        .def_readwrite("schedule", &subproblem::schedule)
        .def_readwrite("size", &subproblem::size)
        .def_readwrite("limit1", &subproblem::limit1)
        .def_readwrite("limit2", &subproblem::limit2)
        ;

    //abstract bounding class
    py::class_<bound_abstract<int>>(m, "_bound_base")
            .def("init", &bound_abstract<int>::init)
            .def("eval", &bound_abstract<int>::evalSolution);

    //expose flowshop bound
    //>>    inst=test_add.instance_taillard('ta20')
    //>>    bound=module.bound_fsp()
    //>>    bound.init(inst)
    //>>    bound.eval(s.schedule)
    py::class_<bound_fsp_weak>(m, "bound_fsp")
        .def(py::init<>())
        .def("init", &bound_fsp_weak::init)
        .def("eval", &bound_fsp_weak::evalSolution)
        ;

    //abstract instance
    py::class_<instance_abstract, std::shared_ptr<instance_abstract>>(m, "_instance_base");
        // .def("init", &instance_abstract::init);

    //expose taillard instance
    py::class_<instance_taillard, instance_abstract, std::shared_ptr<instance_taillard>>(m, "instance_taillard")
        .def(py::init<const char*>())
        .def("get_job_number", &instance_taillard::get_job_number)
        .def("get_machine_number", &instance_taillard::get_machine_number);

    py::class_<pbab>(m, "pbab")
        .def(py::init<>())
        .def(py::init<std::shared_ptr<instance_abstract>>())
        .def("set_initial_solution", py::overload_cast<>(&pbab::set_initial_solution))
        .def("set_initial_solution", py::overload_cast<const std::vector<int>, const int>(&pbab::set_initial_solution))
        .def("print_stats",&pbab::printStats)
        .def_readwrite("inst", &pbab::inst)
    ;



    //###### Flowshop BB
    py::class_<Intervalbb<int>, std::shared_ptr<Intervalbb<int>>>(m,"intervalbb")
        .def(py::init<pbab*>())
        .def("set_root",py::overload_cast<const std::vector<int>>(&Intervalbb<int>::setRoot))
        .def("init_at_interval",&Intervalbb<int>::initAtInterval)
        .def("run",&Intervalbb<int>::run)
    ;

    m.def("make_ivmbb", &make_ivmbb<int>);


    py::class_<VictimSelector, std::shared_ptr<VictimSelector>>(m, "_victim_selector");

    py::class_<ThreadController, std::shared_ptr<ThreadController>>(m, "_thread_controller")
        // .def(py::init<pbab*,int>)
        // .def("set_ws",&ThreadController::set_victim_select)
    ;

    py::class_<IVMController, ThreadController, std::shared_ptr<IVMController>>(m, "IVMController")
        .def(py::init<pbab*,int>())
        .def("run",&IVMController::next)
        .def("set_ws",&IVMController::set_victim_select)
        .def("init_intervals",&IVMController::initFromFac)
    ;

    m.def("make_victim_selector", &make_victim_selector);


    //###### Flowshop Heuristics
    //>>    neh=test_add.fastNEH(inst)
    //>>    neh.run(s)
    py::class_<fastNEH>(m, "fastNEH")
        .def(py::init<instance_abstract&>())
        .def(py::init<const std::vector<std::vector<int>>, const int, const int>())
        .def("run", py::overload_cast<std::shared_ptr<subproblem>>(&fastNEH::run)) //there are 2 run methods
        .def("run", py::overload_cast<std::vector<int>&,int&>(&fastNEH::run))
        .def("run", py::overload_cast<>(&fastNEH::run))
        ;

    py::class_<LocalSearchBase>(m, "_local_search_base")
        .def(py::init<>())
    ;

    py::class_<LocalSearch, LocalSearchBase>(m, "local_search")
        .def(py::init<instance_abstract&>())
        .def(py::init<const std::vector<std::vector<int>>, const int, const int>())
        .def("run_bre", py::overload_cast<std::vector<int>&>(&LocalSearch::localSearchBRE))
        .def("run_bre", py::overload_cast<std::vector<int>&,int,int>(&LocalSearch::localSearchBRE))
        .def("run_ki", &LocalSearch::localSearchKI)
        ;

    py::class_<IG>(m,"ils")
        .def(py::init<instance_abstract&>())
        .def(py::init<const std::vector<std::vector<int>>, const int, const int>())
        .def("run", &IG::run)
        ;

    // py::class_<Beam>(m,"beam")
    //         .def(py::init<instance_abstract&>())
    //         .def(py::init<const std::vector<std::vector<int>>, const int, const int>())
    //         .def("run", &IG::run)
    //         ;


    m.def("printvec", &print_vector, "A function that prints a vector");
}
