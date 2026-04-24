#pragma once
#include"../../Tensor.hpp"
#include"../../Torch.hpp"
#include<vector>
namespace torch{
namespace optim{
template<typename real>
class SGD{
private:
    std::vector<Tensor<real>> params_;
    real lr_;
public:
    SGD(const std::vector<Tensor<real>>& params,real lr):params_(params),lr_(lr){}
    void zero_grad();
    void step();
};

template<typename real>
void SGD<real>::zero_grad(){
    for(Tensor<real>& t : params_){
        t.zero_grad();
    }
}

template<typename real>
void SGD<real>::step(){
    bool prev_grad_status = is_grad_enabled;
    is_grad_enabled = false;
    for(Tensor<real>& param : params_){
        if(!param.is_contiguous()) throw std::runtime_error("assersion error. parameters tensor must be contiguous");

        if(param.get_autograd_meta() && param.get_autograd_meta()->grad.data_ptr() != nullptr){
            real* param_data = param.data_ptr();
            Tensor<real> safe_grad = param.get_autograd_meta()->grad.contiguous();
            if(param.numel() != param.get_autograd_meta()->grad.numel()) throw std::runtime_error("assersion error! a parameter tensor's size must be equal to its grad");

            real* grad_data = safe_grad.data_ptr();
            for(int i=0;i<param.numel();++i){
                param_data[i] = param_data[i] - lr_ * grad_data[i];
            }
        }
    }

    is_grad_enabled = prev_grad_status;
}
}
}