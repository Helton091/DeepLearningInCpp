#pragma once
#include"../../Torch.hpp"
#include"AutoGrad.hpp"

namespace torch{

template<typename real>
class BackwardFunction{
public:
    virtual ~BackwardFunction() = default;
    virtual void apply(const Tensor<real>& grad_output) = 0;
};

template<typename real>
class AddBackward : public BackwardFunction<real> {
private:
    Tensor<real> tensor_a_;
    Tensor<real> tensor_b_;
public:
    AddBackward(const Tensor<real>& a,const Tensor<real>& b):tensor_a_(a),tensor_b_(b){};

};

template<typename real>
bool Tensor<real>::is_leaf(){
    return autograd_meta_ && autograd_meta_->requires_grad;
}

template<typename real>
std::shared_ptr<BackwardFunction<real>> Tensor<real>::grad_fn(){
    return autograd_meta_->grad_fn;
}

template<typename real>
Tensor<real> Tensor<real>::grad(){
    return autograd_meta_->grad;
}


}


