#ifndef TORCH_TENSOR_GEN_HPP_
#define TORCH_TENSOR_GEN_HPP_
#include"../Tensor.hpp"
#include<cmath>
#include<random>
namespace torch{

inline std::mt19937& get_generator(){
    static std::mt19937 generator(std::random_device{}());
    return generator;
}

inline void manual_seed(unsigned int seed){get_generator().seed(seed);}

template<typename real = float>
Tensor<real> rand_like(const Tensor<real>& other,bool requires_grad=false);
template<typename real = float>
Tensor<real> randn_like(const Tensor<real>& other,bool requires_grad=false);
template<typename real = float>
Tensor<real> rand(const std::vector<int>& shape,bool requires_grad=false);

template<typename real = float>
Tensor<real> randn(const std::vector<int>& shape,bool requires_grad=false);

template<typename real = float>
Tensor<real> arange_impl(double start,double end,double step);

template<typename real = float>
Tensor<real> arange(double end);

template<typename real = float>
Tensor<real> arange(double start,double end);

template<typename real = float>
Tensor<real> arange(double start,double end,double step);
template<typename real = float>
Tensor<real> ones(const std::vector<int>& shape,bool requires_grad = false);

template<typename real = float>
Tensor<real> zeros(const std::vector<int>& shape,bool requires_grad = false);

template<typename real = float>
Tensor<real> full(const std::vector<int>& shape,real fill_value,bool requires_grad=false);

template<typename real = float>
Tensor<real> ones_like(const Tensor<real>& other,bool requires_grad=false);

template<typename real = float>
Tensor<real> zeros_like(const Tensor<real>& other,bool requires_grad=false);

template<typename real = float>
Tensor<real> full_like(const Tensor<real>& t,real fill_value,bool requires_grad=false);
}

namespace torch{
template<typename real>
Tensor<real> rand_like(const Tensor<real>& other,bool requires_grad){
    return rand<real>(other.shape(),requires_grad);
}

template<typename real>
Tensor<real> randn_like(const Tensor<real>& other,bool requires_grad){
    return randn<real>(other.shape(),requires_grad);
}

template<typename real>
Tensor<real> rand(const std::vector<int>& shape,bool requires_grad){
    Tensor<real> output(shape,requires_grad);
    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<real> dist(0.0,1.0);
    real* ptr = output.data_ptr();
    for(int i=0;i<output.numel();++i) ptr[i] = dist(gen);
    return output;
}
template<typename real>
Tensor<real> randn(const std::vector<int>& shape,bool requires_grad){
    Tensor<real> output(shape,requires_grad);
    std::mt19937& gen = get_generator();
    std::normal_distribution<real> dist(0.0,1.0);
    real* ptr = output.data_ptr();
    for(int i=0;i<output.numel();++i) ptr[i] = dist(gen);
    return output;
}

template<typename real>
Tensor<real> arange(double start,double end,double step){
    return arange_impl<real>(start,end,step);
}


template<typename real>
Tensor<real> arange(double start,double end){
    return arange_impl<real>(start,end,1.0);
}


template<typename real>
Tensor<real> arange(double end){
    return arange_impl<real>(0.0,end,1.0);
}

template<typename real>
Tensor<real> arange_impl(double start,double end,double step){
    double raw_times = std::ceil((end - start) / step);
    if ((start + (raw_times - 1) * step) >= end && step > 0) raw_times -= 1;
    if ((start + (raw_times - 1) * step) <= end && step < 0) raw_times -= 1;
    int times = std::max(0, static_cast<int>(raw_times));
    std::vector<int> shape(1,times);
    Tensor<real> result(shape);
    real* real_ptr = result.data_ptr();
    for(int i=0;i<times;++i){
        real_ptr[i] = static_cast<real>(start) + static_cast<real>(i*step);
    }
    return result;
}

template<typename real>
Tensor<real> full_like(const Tensor<real>& t,real fill_value,bool requires_grad){
    Tensor<real> output(t.shape(),requires_grad);
    output.fill_(fill_value);
    return output;
}


template<typename real>
Tensor<real> full(const std::vector<int>& shape,real fill_value,bool requires_grad){
    Tensor<real> t(shape,requires_grad);
    t.fill_(fill_value);
    return t;
}
template<typename real>
Tensor<real> ones_like(const Tensor<real>& other,bool requires_grad){
    Tensor<real> t(other.shape(),requires_grad);
    t.fill_(static_cast<real>(1));
    return t;
}

template<typename real>
Tensor<real> zeros_like(const Tensor<real>& other,bool requires_grad){
    Tensor<real> t(other.shape(),requires_grad);
    t.fill_(static_cast<real>(0));
    return t;
}

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