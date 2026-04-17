#pragma once
#include"../../Torch.hpp"
namespace torch{

template<typename real = float>
struct AutogradMeta{
    Tensor<real> grad;
    bool has_grad = false;
    std::shared_ptr<BackwardFunction<real>> grad_fn = nullptr;
    bool is_lead = true;
    bool requires_grad = false;
};
}
