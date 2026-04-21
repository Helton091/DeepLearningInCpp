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
    AddBackward(const Tensor<real>& a,const Tensor<real>& b):tensor_a_(a),tensor_b_(b){}
    void apply(const Tensor<real>& grad_output) override;
};

template<typename real>
class AddBackwardScaler : public BackwardFunction<real> {
private:
    Tensor<real> tensor;
public:
    AddBackwardScaler(const Tensor<real>& a):tensor(a){}
    void apply(const Tensor<real>& grad_output) override;
};

template<typename real>
void AddBackwardScaler<real>::apply(const Tensor<real>& grad_output){
    if(tensor.requires_grad()){
        tensor.add_grad(grad);
    }
}

template<typename real>
void AddBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_a_.requires_grad()){
        Tensor<real> grad = unbroadcast(grad_output,tensor_a_.shape());
        tensor_a_.add_grad(grad);
    }
    if(tensor_b_.requires_grad()){
        Tensor<real> grad = unbroadcast(grad_output,tensor_b_.shape());
        tensor_b_.add_grad(grad);
    }
}
template<typename real>
bool Tensor<real>::is_leaf() const {
    // If it requires grad but has no grad_fn (creator), it is a leaf.
    // Alternatively, just return the is_leaf flag from meta.
    return autograd_meta_ && autograd_meta_->is_leaf;
}

template<typename real>
std::shared_ptr<BackwardFunction<real>> Tensor<real>::grad_fn() const {
    if (!autograd_meta_) return nullptr;
    return autograd_meta_->grad_fn;
}

template<typename real>
Tensor<real> Tensor<real>::grad() const {
    if (!autograd_meta_) throw std::runtime_error("Tensor does not require grad");
    return autograd_meta_->grad;
}

template<typename real>
void Tensor<real>::set_grad_fn(std::shared_ptr<BackwardFunction<real>> fn) {
    if (autograd_meta_) {
        autograd_meta_->grad_fn = fn;
        autograd_meta_->is_leaf = false; // It has a creator now
    }
}

template<typename real>
void Tensor<real>::add_grad(const Tensor<real>& g) {
    if (!autograd_meta_) return;
    
    if (!autograd_meta_->has_grad) {
        autograd_meta_->grad = g;
        autograd_meta_->has_grad = true;
    } else {
        autograd_meta_->grad = autograd_meta_->grad + g;
    }
}

template<typename real>
void Tensor<real>::backward() {
    // Empty placeholder for the backward engine (Step 4)
    std::cout << "Backward pass triggered!\n";
}

}


