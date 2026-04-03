#include"Tensor.hpp"
#include<iostream>
using std::cout,std::cin,std::endl;
int main(){
    Tensor<float> t(std::vector(3,2));
    Tensor<float> s(std::vector(2,2));
    cout << t << endl << s << endl << t+s;

    return 0;
}