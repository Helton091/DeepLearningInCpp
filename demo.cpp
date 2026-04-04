#include"Tensor.hpp"
#include<iostream>
using std::cout,std::cin,std::endl;
using namespace torch;
int main(){
    Tensor<float> t(std::vector{3,2});
    Tensor<float> s(std::vector{3,1,2});
    t.fill_(1);
    s.fill_(0.5);
    cout << t << endl << s << endl << t+s;

    return 0;
}