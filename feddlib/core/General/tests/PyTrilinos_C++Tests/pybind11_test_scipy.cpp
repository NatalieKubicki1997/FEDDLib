#include <iostream>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>           // mandatory for myPyObject.cast<std::vector<T>>()
#include <pybind11/functional.h>    // mandatory for py::cast( std::function )

namespace py = pybind11;

int main(int argc, char *argv[]) {
        py::scoped_interpreter guard{};

        py::module np = py::module::import("numpy");
        py::object random = np.attr("random");

        py::module scipy = py::module::import("scipy.optimize");

        // create some data for fitting
        std::vector<double> xValues(11, 0);
        std::vector<double> yValues(11, 0);
        for (int i = -5; i < 6; ++i) {
            xValues[i + 5] = i;
            yValues[i + 5] = i*i;
        }

        // cast it to numpy arrays
        py::array_t<double> pyXValues = py::cast(xValues);
        py::array_t<double> pyYValues = py::cast(yValues);

        // add some noise to the yValues using numpy -> Works!
        py::array_t<double> pyYValuesNoise = np.attr("add")(pyYValues, random.attr("randn")(11));

        // create a function f_a(x) = a*x^2
        std::function<std::vector<double>(std::vector<double>, double)> squared = [](std::vector<double> x, double a) {
            std::vector<double> retvals(x);
            std::transform(x.begin(), x.end(), retvals.begin(), [a](double val) { return a*val*val; });
            return retvals;
        };

        // cast it to a python function
        py::function pySquared = py::cast(squared);     

// In this code, we create the objective function in Python using py::exec. Then, we retrieve it as objective_function and pass it to curve_fit. 
//This way, you are providing a valid Python function to curve_fit, and it should work correctly.
/*py::str objective_code = R"(
def objective(x, a):
    return a * x**2
        )";

        py::object main_module = py::module::import("__main__");
        py::object objective_namespace = main_module.attr("__dict__");
        py::exec(objective_code, objective_namespace);
        py::object objective_function = objective_namespace["objective"];
*/

        // get scipy.optimize.curve_fit
        py::function curve_fit = scipy.attr("curve_fit");
/*
        // call curve_fit -> throws exception
        /* py::object = curve_fit(pySquared, pyXValues, pyYValues);
        */
       

        // See https://stackoverflow.com/questions/51762140/using-scipy-from-c-via-pybind11
        std::cout << "It worked" << std::endl;

    return 0;
}

 
/*

using namespace Teuchos;

int main(int argc, char *argv[]) {

    RCP<int> test(new int);
    *test = 1;    

    return(EXIT_SUCCESS);
}
*/