#include "library.h"

#include <iostream>

using namespace ct;

void MyExportedClass::exported_method() {
    std::cout << "Hello from exported method" << std::endl;
}

int ct::exported_function(int arg) {
    return 2 * arg;
}