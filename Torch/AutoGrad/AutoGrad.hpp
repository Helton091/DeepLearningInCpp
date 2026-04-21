#pragma once
#include"../../Torch.hpp"
namespace torch{

template<typename real = float>
struct AutogradMeta{
    Tensor<real> grad;
    bool has_grad = false;
    std::shared_ptr<BackwardFunction<real>> grad_fn = nullptr;
    bool is_leaf = true;
    bool requires_grad = false;
};

template<typename real>
Tensor<real> unbroadcast(const Tensor<real>& grad_output,const std::vector<int> target_shape);
}

namespace torch{
template<typename real>
Tensor<real> unbroadcast(const Tensor<real>& grad_output,const std::vector<int> target_shape){
    int grad_output_ndim = grad_output.shape().size();
    int target_ndim = target_shape.size();
    if(grad_output_ndim < target_ndim) throw std::out_of_range("when unbroadcasting, ndim of grad_output < ndim of target tensor");
    Tensor<real> output = grad_output;
    if(grad_output_ndim > target_ndim){
        for(int i=0;i<grad_output_ndim - target_ndim;++i) output = output.sum(0,false);
    }
    for(int i=0;i<target_ndim;++i){
        if(target_shape[i]!=1 && target_shape[i] != output.shape()[i]){
            throw std::runtime_error("shape mismatch when unbroadcasting");
        } else if(target_shape[i]==1 && output.shape()[i] > 1){
            output = output.sum(i,true);
        }
    }
    return output;

}

}
