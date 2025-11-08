module;

#include <iostream>

module module_example;


void ct::exported_fn(int a) {
    std::cout << a << std::endl;
}


int another_exported_fn(int b) {
    std::cout << b << std::endl;
    return b;
}


struct MyStruct {
    int d_;
    [[nodiscard]] int get_d() const { return d_; }
};