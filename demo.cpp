#include"Tensor.hpp"
#include<iostream>
using std::cout,std::cin;
int main(){
    Tensor<float> t(std::vector(3,2));
    cout << t;

    return 0;
}