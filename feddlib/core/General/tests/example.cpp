#include <Teuchos_RCP.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/*!

 @brief  pybind11 test - simple test case to try pybind11 with binder for including RCP
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */

using namespace Teuchos;

int add(int i, int j) {
        return i + j;
}
/*
std::vector<int> get_a_vector(){
     std::vector<int> a_vector;
     a_vector = std::vector<int>{1, 2, 3, 4, 5};
    return a_vector;
    }
*/

/*void return_rcp()
{
    RCP<int> test(new int);
    *test = 1;
    //return test;
}
*/

PYBIND11_MODULE(example, m) {
       // py::class_<RCP<int>>(m, "RCP")
       // .def(py::init<int*>());


        m.doc() = "pybind11 example plugin for rcp return"; // optional module docstring
        //m.def("get_a_vector", &get_a_vector, "A function that return rcp");
        m.def("add", &add, "A function that adds two numbers");
        //m.def("return_rcp", &return_rcp, "A function that return rcp");



}



