#ifndef TORCH_TENSOR_GEN_HPP_
#define TORCH_TENSOR_GEN_HPP_
#include"Tensor.hpp"
namespace torch{
    template<typename real>
    Tensor<real> ones(const std::vector<int>& shape,bool requires_grad = false);

    template<typename real>
    Tensor<real> zeros(const std::vector<int>& shape,bool requires_grad = false);
}

namespace torch{
    template<typename real>
    Tensor<real> ones(const std::vector<int>& shape,bool requires_grad){
        Tensor<real> t(shape, requires_grad);
        t.fill_(static_cast<real>(1));
        return t;
    }

    template<typename real>
    Tensor<real> zeros(const std::vector<int>& shape,bool requires_grad){
        Tensor<real> t(shape, requires_grad);
        t.fill_(static_cast<real>(0));
        return t;
    }

}
#endif