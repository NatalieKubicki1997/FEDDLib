#include <Teuchos_RCP.hpp>

/*!

 @brief  pybind11 test - simple test case to try pybind11 with binder for including RCP
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */


using namespace Teuchos;

int main(int argc, char *argv[]) {

    RCP<int> test(new int);
    *test = 1;    

    return(EXIT_SUCCESS);
}
