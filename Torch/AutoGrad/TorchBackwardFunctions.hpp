#pragma once
#include"../../Torch.hpp"
#include"AutoGrad.hpp"

namespace torch{

template<typename real>
class BackwardFunction{
public:
    virtual ~BackwardFunction() = default;
    virtual void apply(const Tensor<real>& grad_output) = 0;
    virtual std::vector<Tensor<real>> get_inputs() const = 0;
};
template<typename real>
class DivBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_a_;
    Tensor<real> tensor_b_;
public:
    DivBackward(const Tensor<real>& a,const Tensor<real>& b) : tensor_a_(a),tensor_b_(b){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class MulBackwardScaler : public BackwardFunction<real>{
private:
    Tensor<real> tensor_a_;
    real b_;
public:
    MulBackwardScaler(const Tensor<real>& a, real b);
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class MulBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_a_;
    Tensor<real> tensor_b_;
public:
    MulBackward(const Tensor<real>& a,const Tensor<real>& b);
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class SubBackwardScaler : public BackwardFunction<real>{
private:
    bool is_a_first;
    Tensor<real> tensor_a_;
public:
    SubBackwardScaler(const Tensor<real>& a,bool is_a_first);
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class SubBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_a_;
    Tensor<real> tensor_b_;
public:
    SubBackward(const Tensor<real>& a,const Tensor<real>& b);
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class AddBackward : public BackwardFunction<real> {
private:
    Tensor<real> tensor_a_;
    Tensor<real> tensor_b_;
public:
    AddBackward(const Tensor<real>& a,const Tensor<real>& b):tensor_a_(a),tensor_b_(b){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};


template<typename real>
class AddBackwardScaler : public BackwardFunction<real> {
private:
    Tensor<real> tensor;
public:
    AddBackwardScaler(const Tensor<real>& a):tensor(a){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};
template<typename real>
class DivBackwardScaler : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    real num_;
    bool is_tensor_first;
public:
    DivBackwardScaler(const Tensor<real>& t,real n,bool is_t_first) : tensor_(t),num_(n),is_tensor_first(is_t_first){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};
template<typename real>
std::vector<Tensor<real>> SubBackwardScaler<real>::get_inputs() const{
    return {tensor_a_};
}

template<typename real>
std::vector<Tensor<real>> DivBackwardScaler<real>::get_inputs() const{
    return {tensor_};
}
template<typename real>
void DivBackwardScaler<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        if(is_tensor_first){
            Tensor<real> new_grad = grad_output / num_;
            tensor_.add_grad(new_grad);
        } else {
            Tensor<real> new_grad = (grad_output * (num_ * static_cast<real>(-1.0))) / (tensor_ * tensor_);
            tensor_.add_grad(new_grad);
        }
    }
}

template<typename real>
std::vector<Tensor<real>> DivBackward<real>::get_inputs() const{
    return {tensor_a_,tensor_b_};
}

template<typename real>
void DivBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_a_.requires_grad()){
        tensor_a_.add_grad(unbroadcast(grad_output / tensor_b_, tensor_a_.shape()));
    }
    if(tensor_b_.requires_grad()){
        Tensor<real> new_grad = (grad_output * tensor_a_) / tensor_b_;
        new_grad = new_grad / tensor_b_;
        new_grad = new_grad * static_cast<real>(-1.0);
        tensor_b_.add_grad(unbroadcast(new_grad, tensor_b_.shape()));
    }
}


template<typename real>
void MulBackwardScaler<real>::apply(const Tensor<real>& grad_output){
    if(tensor_a_.requires_grad()){
        tensor_a_.add_grad(grad_output * b_);
    }
}

template<typename real>
std::vector<Tensor<real>> MulBackward<real>::get_inputs() const{
    return {tensor_a_,tensor_b_};
}

template<typename real>
void MulBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_a_.requires_grad()){
        tensor_a_.add_grad(unbroadcast(grad_output * tensor_b_,tensor_a_.shape()));
    }
    if(tensor_b_.requires_grad()){
        tensor_b_.add_grad(unbroadcast(grad_output * tensor_a_,tensor_b_.shape()));
    }

}

template<typename real>
MulBackwardScaler<real>::MulBackwardScaler(const Tensor<real>& a, real b) : tensor_a_(a), b_(b){}

template<typename real>
MulBackward<real>::MulBackward(const Tensor<real>& a,const Tensor<real>& b) : tensor_a_(a),tensor_b_(b){}

template<typename real>
std::vector<Tensor<real>> MulBackwardScaler<real>::get_inputs() const{
    return {tensor_a_};
}

template<typename real>
void SubBackwardScaler<real>::apply(const Tensor<real>& grad_output){
    if(tensor_a_.requires_grad()){
        if(is_a_first){
            // C = A - scalar, dC/dA = 1
            tensor_a_.add_grad(grad_output);
        } else {
            // C = scalar - A, dC/dA = -1
            tensor_a_.add_grad(grad_output * static_cast<real>(-1.0));
        }
    }
}

template<typename real>
void SubBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_a_.requires_grad()){
        tensor_a_.add_grad(unbroadcast(grad_output,tensor_a_.shape()));
    }
    if(tensor_b_.requires_grad()){
        Tensor<real> grad = grad_output * static_cast<real>(-1.0);
        tensor_b_.add_grad(unbroadcast(grad,tensor_b_.shape()));
    }
}

template<typename real>
SubBackwardScaler<real>::SubBackwardScaler(const Tensor<real>& a,bool is_a_first) : tensor_a_(a),is_a_first(is_a_first){}

template<typename real>
SubBackward<real>::SubBackward(const Tensor<real>& a,const Tensor<real>& b): tensor_a_(a),tensor_b_(b){}



template<typename real>
std::vector<Tensor<real>> SubBackward<real>::get_inputs() const{
    return {tensor_a_,tensor_b_};
}

template<typename real>
std::vector<Tensor<real>> AddBackward<real>::get_inputs() const{
    return {tensor_a_,tensor_b_};
}

template<typename real>
std::vector<Tensor<real>> AddBackwardScaler<real>::get_inputs() const {
    return {tensor};
}

template<typename real>
void AddBackwardScaler<real>::apply(const Tensor<real>& grad_output){
    if(tensor.requires_grad()){
        tensor.add_grad(grad_output);
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
    if (!autograd_meta_->has_grad) throw std::runtime_error("Tensor has no grad yet (has_grad is false)");
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
    std::unordered_map<AutogradMeta<real>*,int> indegree;
    std::unordered_set<AutogradMeta<real>*> visited;
    std::queue<Tensor<real>> ready_queue;
    std::queue<Tensor<real>> bfs_queue;
    is_grad_enabled = false;
    //step1 : calculate indegree for each autograd_meta
    bfs_queue.push(*this);
    AutogradMeta<real>* first_meta = this->get_autograd_meta();
    visited.insert(first_meta);
    while(!bfs_queue.empty()){
        Tensor<real> curr_tensor = bfs_queue.front();
        bfs_queue.pop();
        AutogradMeta<real>* curr_meta = curr_tensor.get_autograd_meta();
        BackwardFunction<real>* curr_backward_fn = curr_meta->grad_fn.get();
        if(curr_backward_fn != nullptr) {
            std::vector<Tensor<real>> next_tensors = curr_backward_fn->get_inputs();
            for(Tensor<real> t : next_tensors){
                AutogradMeta<real>* next_meta = t.get_autograd_meta();
                if(!visited.count(next_meta)){
                    bfs_queue.push(t);
                    visited.insert(next_meta);
                }
                ++indegree[next_meta];
            }
        }
    }
    //step2 : 
    ready_queue.push(*this);
    
    if(first_meta->has_grad == false){
        first_meta->grad = ones_like(*this,false);
    }
    while(!ready_queue.empty()){
        Tensor<real> curr_tensor = ready_queue.front();
        ready_queue.pop();

        AutogradMeta<real>* curr_meta = curr_tensor.get_autograd_meta();
        BackwardFunction<real>* back_fn = curr_meta->grad_fn.get();
        if(back_fn != nullptr){
            back_fn->apply(curr_meta->grad);
            std::vector<Tensor<real>> next_tensors = back_fn->get_inputs();
            for(Tensor<real> t : next_tensors){
                AutogradMeta<real>* next_meta = t.get_autograd_meta();
                if(--indegree[next_meta] == 0) ready_queue.push(t);
            }
        }
    }
    std::cout << "Backward pass triggered!\n";
    is_grad_enabled = true;
}

}


