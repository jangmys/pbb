#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "add.h"
#include "libbounds.h"
#include "subproblem.h"

#include "matrix_controller.h"


#include "libheuristic.h"

namespace py = pybind11;
// #include "../evaluation/libbounds.h"
// #include "../heuristics/flowshop/neh/fastNEH.h"

// int add(int i, int j) {
//     return i + j;
// }

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



PYBIND11_MODULE(test_add, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("adder", &adder, "A function that adds two numbers",
          py::arg("i") = 1, py::arg("j") = 2
    );

    m.attr("the_answer") = 42;

    py::object world = py::cast("World");
    m.attr("what") = world;

    py::class_<test>(m, "test")
        .def(py::init<>())
        .def("init", &test::init);

    //=============================================================

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
    // .def("init", &subproblem::init);
    //     .def("eval", &bound_fsp_weak::evalSolution);


    // py::class_<bound_fsp_weak, bound_abstract<int>>(m, "bound_fsp")
    //     .def("init", &bound_fsp_weak::init);

    py::class_<bound_abstract<int>>(m, "_bound_base")
            .def("init", &bound_abstract<int>::init)
            .def("eval", &bound_abstract<int>::evalSolution);

    py::class_<bound_fsp_weak>(m, "bound_fsp")
        .def(py::init<>())
        .def("init", &bound_fsp_weak::init)
        .def("eval", &bound_fsp_weak::evalSolution)
        ;

    py::class_<instance_abstract>(m, "_instance_base");
        // .def("init", &instance_abstract::init);

    py::class_<instance_taillard, instance_abstract>(m, "instance_taillard")
        .def(py::init<const char*>())
        .def("get_job_number", &instance_taillard::get_job_number)
        .def("get_machine_number", &instance_taillard::get_machine_number);

    // py::class_<pbab>(m, "pbab")
    //     .def(py::init<>())
    //     .def(py::init<std::shared_ptr<instance_abstract>>())
    //     ;
    //
    // py::class_<matrix_controller>(m, "matrix_controller")
    //     .def(py::init<pbab*,int,bool>());

    py::class_<fastNEH>(m, "fastNEH")
        .def(py::init<instance_abstract&>())
        .def(py::init<const std::vector<std::vector<int>>, const int, const int>())
        // .def("run", py::overload_cast<std::vector<int>&, int &>(&fastNEH::run))
        .def("run", py::overload_cast<std::shared_ptr<subproblem>>(&fastNEH::run))
        ;


    m.def("printvec", &print_vector, "A function that prints a vector");
}
