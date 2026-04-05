#include"Torch.hpp"
#include<iostream>
using std::cout,std::cin,std::endl;
using namespace torch;
int main(){
    Tensor<float> t = torch::randn<float>(std::vector{3,2,2});
    
    return 0;
}