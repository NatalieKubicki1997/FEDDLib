file(GLOB TESTFILES *.npy *.txt *.py *.yaml )

file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)
add_test(PyTrilinos_Test ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Test_Trilinos_Pybind.py")
set_tests_properties(PyTrilinos_Test PROPERTIES ENVIRONMENT "${PYTHONPATH}")


SET(FEDDLIB_PYTHON_N_MPI 2)
add_test(PyTrilinos_Test_Parallel "/usr/bin/mpiexec" "-np" "${FEDDLIB_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Test_Trilinos_Pybind_MPI.py")
set_tests_properties(PyTrilinos_Test PROPERTIES ENVIRONMENT "${PYTHONPATH}")


file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)
add_test(PyTrilinos_Test ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/GaussianProcessRegression.py")
set_tests_properties(PyTrilinos_Test PROPERTIES ENVIRONMENT "${PYTHONPATH}")