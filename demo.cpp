#include"Torch.hpp"
#include<iostream>
using std::cout,std::cin,std::endl;
using namespace torch;
int main(){
    Tensor<float> t(std::vector{3,1,2});
    Tensor<float> s(std::vector{3,2});
    Tensor<float> t1 = torch::arange<float>(3);
    Tensor<float> t2 = torch::arange<float>(1,3);
    Tensor<float> t3 = torch::arange<float>(0,-3,-1);
    cout << t1 << endl << t2 << endl << t3;

    return 0;
}