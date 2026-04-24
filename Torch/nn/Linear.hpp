#pragma once
#include"../../Tensor.hpp"
#include"../../Torch.hpp"
#include"Module.hpp"
namespace torch{
namespace nn{
template<typename real>
class Linear : public Module<real>{
private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    Tensor<real> weight_;
    Tensor<real> bias_;
public:
    Linear(int in_features,int out_features,bool use_bias = true) : in_features_(in_features),out_features_(out_features),use_bias_(use_bias){
        weight_ = this->register_parameter("weight", torch::ones<real>({in_features_, out_features_}, true));
        if(use_bias_) {
            bias_ = this->register_parameter("bias", torch::ones<real>({out_features_}, true));
        }
    }
    Tensor<real> forward(const Tensor<real>& x) override{
        if(!use_bias_){
            Tensor<real> x1 = torch::matmul(x, weight_); 
            return x1;
        }
        else{
            Tensor<real> x1 = torch::matmul(x, weight_) + bias_; 
            return x1;
        }
    }
};

}
}