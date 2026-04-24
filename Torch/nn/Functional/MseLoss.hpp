#pragma once
#include"../../../Tensor.hpp"
#include"../../../Torch.hpp"
namespace torch{
namespace nn{
namespace functional{
template<typename real>
Tensor<real> mse_loss(const Tensor<real>& pred, const Tensor<real>& target){
    Tensor<real> output = (pred - target) * (pred - target);
    output = output.sum() / static_cast<real>(output.numel());
    return output;
}

}
}
}