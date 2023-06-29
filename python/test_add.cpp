#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "add.h"
#include "libbounds.h"

// #include "../evaluation/libbounds.h"
// #include "../heuristics/flowshop/neh/fastNEH.h"

// int add(int i, int j) {
//     return i + j;
// }


PYBIND11_MODULE(test_add, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("adder", &adder, "A function that adds two numbers",
          py::arg("i") = 1, py::arg("j") = 2
    );

    m.attr("the_answer") = 42;

    py::object world = py::cast("World");
    m.attr("what") = world;

    py::class_<bound_abstract<int>>(m, "_bound_base")
            .def("init", &bound_abstract<int>::init);

    // py::class_<bound_fsp_weak, bound_abstract<int>>(m, "bound_fsp")
    //     .def("init", &bound_fsp_weak::init);

}
