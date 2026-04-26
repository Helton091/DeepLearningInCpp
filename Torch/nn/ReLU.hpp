#pragma once
#include"../../Torch.hpp"
#include"../../Tensor.hpp"
#include"../../Tensor/TensorUnaryOps.hpp"
namespace torch{
namespace nn{
template<typename real>
class ReLU : public Module<real>{
public:
    Tensor<real> forward(const Tensor<real>& x) override { return x.relu(); }
};
    
}
}