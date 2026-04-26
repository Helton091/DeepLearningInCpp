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
class MatMulBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_a_;
    Tensor<real> tensor_b_;
public:
    MatMulBackward(const Tensor<real>& a,const Tensor<real>& b) : tensor_a_(a),tensor_b_(b){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class TransposeBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    int dim0;
    int dim1;
public:
    TransposeBackward(const Tensor<real>& t,int d0,int d1) : tensor_(t),dim0(d0),dim1(d1){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class PermuteBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    std::vector<int> changed_dim_back;
public:
    PermuteBackward(const Tensor<real>& tensor_,const std::vector<int>& changed_dim_origin);
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};
template<typename real>
class ReshapeBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    std::vector<int> original_shape_;
public:
    ReshapeBackward(const Tensor<real>& t,const std::vector<int>& original_shape) : tensor_(t),original_shape_(original_shape){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class SqueezeBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    int dim_;
public:
    SqueezeBackward(const Tensor<real>& tensor,int dim) : tensor_(tensor),dim_(dim){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class SqueezeAllBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    std::vector<int> original_shape_;
public:
    SqueezeAllBackward(const Tensor<real>& tensor,const std::vector<int> original_shape) : tensor_(tensor),original_shape_(original_shape){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};
template<typename real>
class UnSqueezeBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    int dim_;
public:
    UnSqueezeBackward(const Tensor<real>& tensor,int dim) : tensor_(tensor),dim_(dim){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class ContiguousBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
public:
    ContiguousBackward(const Tensor<real>& tensor) : tensor_(tensor){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class SumBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    int dim_;
    bool keep_dim_;
    std::vector<int> original_shape_;
public:
    SumBackward(const Tensor<real>& tensor,int dim,bool keep_dim,const std::vector<int>& original_shape) 
        : tensor_(tensor),dim_(dim),keep_dim_(keep_dim),original_shape_(original_shape){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class ExpandBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
public:
    ExpandBackward(const Tensor<real>& tensor) : tensor_(tensor){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};

template<typename real>
class SumAllBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
    std::vector<int> original_shape_;
public:
    SumAllBackward(const Tensor<real>& tensor,std::vector<int> original_shape) : tensor_(tensor),original_shape_(original_shape){}
    void apply(const Tensor<real>& grad_output) override;
    std::vector<Tensor<real>> get_inputs() const override;
};
template<typename real>
std::vector<Tensor<real>> SumAllBackward<real>::get_inputs() const{
    return {tensor_};
}

template<typename real>
void SumAllBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(grad_output.expand(original_shape_));
    }
}

template<typename real>
std::vector<Tensor<real>> ExpandBackward<real>::get_inputs() const{
    return {tensor_};
}

template<typename real>
std::vector<Tensor<real>> TransposeBackward<real>::get_inputs() const{
    return {tensor_};
}
template<typename real>
void ExpandBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(unbroadcast(grad_output,tensor_.shape()));
    }
}

template<typename real>
std::vector<Tensor<real>> SumBackward<real>::get_inputs() const{
    return {tensor_};
}

template<typename real>
void SumBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        if(keep_dim_){
            tensor_.add_grad(grad_output.expand(original_shape_));
        } else {
            tensor_.add_grad(grad_output.unsqueeze(dim_).expand(original_shape_));
        }   
    }
}

template<typename real>
std::vector<Tensor<real>> ContiguousBackward<real>::get_inputs() const{
    return {tensor_};
}
template<typename real>
void ContiguousBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(grad_output);
    }
}

template<typename real>
std::vector<Tensor<real>> UnSqueezeBackward<real>::get_inputs() const{
    return {tensor_};
}
template<typename real>
void UnSqueezeBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(grad_output.squeeze(dim_));
    }
}

template<typename real>
std::vector<Tensor<real>> SqueezeAllBackward<real>::get_inputs() const{
    return {tensor_};
}

template<typename real>
void SqueezeAllBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(grad_output.reshape(original_shape_));
    }
}

template<typename real>
std::vector<Tensor<real>> SqueezeBackward<real>::get_inputs() const{
    return {tensor_};
}

template<typename real>
void SqueezeBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(grad_output.unsqueeze(dim_));
    }
}

template<typename real>
std::vector<Tensor<real>> ReshapeBackward<real>::get_inputs() const{
    return {tensor_};
}

template<typename real>
void ReshapeBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(grad_output.reshape(original_shape_));
    }
}

template<typename real>
PermuteBackward<real>::PermuteBackward(const Tensor<real>& tensor,const std::vector<int>& changed_dim_origin) : tensor_(tensor){
    int ndim = changed_dim_origin.size();
    changed_dim_back.resize(ndim);
    for(int i=0;i<ndim;++i){
        changed_dim_back[changed_dim_origin[i]] = i;
    }
}
template<typename real>
std::vector<Tensor<real>> PermuteBackward<real>::get_inputs() const{
    return {tensor_};
}

template<typename real>
void PermuteBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(grad_output.permute(changed_dim_back));
    }
}

template<typename real>
void TransposeBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_.requires_grad()){
        tensor_.add_grad(grad_output.transpose(dim0,dim1));
    }
}
template<typename real>
std::vector<Tensor<real>> MatMulBackward<real>::get_inputs() const{
    return {tensor_a_,tensor_b_};
}

template<typename real>
void MatMulBackward<real>::apply(const Tensor<real>& grad_output){
    if(tensor_a_.requires_grad()){
        tensor_a_.add_grad(unbroadcast(matmul(grad_output,tensor_b_.transpose(-2,-1)),tensor_a_.shape()));
    }
    if(tensor_b_.requires_grad()){
        tensor_b_.add_grad(unbroadcast(matmul(tensor_a_.transpose(-2,-1),grad_output),tensor_b_.shape()));
    }
}


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
    is_grad_enabled = true;
}

template<typename real>
class CloneBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_;
public:
    CloneBackward(const Tensor<real>& tensor) : tensor_(tensor){}
    void apply(const Tensor<real>& grad_output) override{
        if(tensor_.requires_grad()) tensor_.add_grad(grad_output);
    }
    std::vector<Tensor<real>> get_inputs() const override{
        return {tensor_};
    }
};

template<typename real>
class ReluBackward : public BackwardFunction<real>{
private:
    Tensor<real> tensor_; //output_tensor, must be contiguous
public:
    ReluBackward(const Tensor<real>& output_tensor) : tensor_(output_tensor){}
    void apply(const Tensor<real>& grad_output) override{
        if(tensor_.requires_grad()){
            Tensor<real> new_grad = grad_output.clone();
            const real* data = tensor_.data_ptr();
            real* grad_data = new_grad.data_ptr();
            for(int i=0;i<tensor_.numel();++i){
                if(data[i] <= 0) grad_data[i] = 0; 
            }
            tensor_.add_grad(new_grad);
        }
    }
    std::vector<Tensor<real>> get_inputs() const override{
        return {tensor_};
    }
};

template<typename real>
class CrossEntropyBackward : public BackwardFunction<real>{
private:
    Tensor<real> logits_;
    Tensor<real> softmax_probs_;
    Tensor<real> targets_;
public:
    CrossEntropyBackward(const Tensor<real>& logits,const Tensor<real>& softmax_probs,const Tensor<real>& targets) : logits_(logits),softmax_probs_(softmax_probs),targets_(targets){}
    void apply(const Tensor<real>& grad_output) override{
        if(!logits_.requires_grad()) return;
        Tensor<real> grad_logits = softmax_probs_.clone();
        real* grad_logits_data = grad_logits.data_ptr();
        const real* targets_data = targets_.data_ptr();

        int batch_size = logits_.shape()[0];
        int num_classes = logits_.shape()[1];

        for(int i=0;i<batch_size;++i){
            int target_class = static_cast<int>(targets_data[i]);
            grad_logits_data[i*num_classes + target_class] -= static_cast<real>(1.0);
        }
        grad_logits = grad_logits / static_cast<real>(batch_size);
        grad_logits = grad_logits * grad_output;
        logits_.add_grad(grad_logits);
    }
    std::vector<Tensor<real>> get_inputs() const override{
        return {logits_};
    }
};

template<typename real>
class Conv2dBackward : public BackwardFunction<real> {
private:
    Tensor<real> X_;
    Tensor<real> W_;
    Tensor<real> bias_;
    Tensor<real> X_col_; // Cached from forward pass
    int stride_;
    int padding_;

public:
    Conv2dBackward(const Tensor<real>& X, const Tensor<real>& W, const Tensor<real>& bias, 
                   const Tensor<real>& X_col, int stride, int padding) 
        : X_(X), W_(W), bias_(bias), X_col_(X_col), stride_(stride), padding_(padding) {}

    void apply(const Tensor<real>& grad_output) override {
        // grad_output is [Batch_size, C_out, H_out, W_out]
        const int C_out = W_.shape()[0];
        const int K_H = W_.shape()[2];
        const int K_W = W_.shape()[3];
        const int X_ndim = X_.shape().size();
        const int C_in_X = X_.shape()[X_ndim - 3];

        int Batch_size = 1;
        for(int i = 0; i < X_ndim - 3; ++i) Batch_size *= X_.shape()[i];
        
        int H_out = grad_output.shape()[grad_output.shape().size() - 2];
        int W_out = grad_output.shape()[grad_output.shape().size() - 1];

        // 1. Reshape grad_output to Y_col format: [Batch_size, C_out, H_out * W_out]
        Tensor<real> grad_Y_col = grad_output.reshape({Batch_size, C_out, H_out * W_out});

        // 2. Gradient w.r.t Bias (if requires_grad)
        if (bias_.numel() > 0 && bias_.requires_grad()) {
            // grad_bias = sum over Batch, H_out, W_out
            // Summing over last dimension (H_out * W_out) -> [Batch_size, C_out]
            Tensor<real> grad_bias = grad_Y_col.sum(-1, false); 
            // Summing over Batch_size -> [C_out]
            if (Batch_size > 1) grad_bias = grad_bias.sum(0, false);
            
            bias_.add_grad(unbroadcast(grad_bias, bias_.shape()));
        }

        // 3. Gradient w.r.t Weight W
        if (W_.requires_grad()) {
            // dW_row = grad_Y_col @ X_col^T
            // grad_Y_col: [Batch, C_out, H_out*W_out]
            // X_col_: [Batch, C_in*K_H*K_W, H_out*W_out] -> transpose(-2,-1) -> [Batch, H_out*W_out, C_in*K_H*K_W]
            Tensor<real> dW_batched = matmul(grad_Y_col, X_col_.transpose(-2, -1));
            // dW_batched: [Batch, C_out, C_in*K_H*K_W]
            
            // Sum over batch dimension to get total weight gradient
            Tensor<real> dW_row = (Batch_size > 1) ? dW_batched.sum(0, false) : dW_batched;
            
            // Reshape back to [C_out, C_in, K_H, K_W]
            Tensor<real> grad_W = dW_row.reshape(W_.shape());
            W_.add_grad(unbroadcast(grad_W, W_.shape()));
        }

        // 4. Gradient w.r.t Input X
        if (X_.requires_grad()) {
            // dX_col = W_row^T @ grad_Y_col
            // W_row: [C_out, C_in*K_H*K_W] -> transpose -> [C_in*K_H*K_W, C_out]
            Tensor<real> W_row = W_.reshape({C_out, C_in_X * K_H * K_W});
            // W_row^T is 2D, grad_Y_col is 3D batched. matmul will broadcast W_row^T to all batches
            Tensor<real> dX_col = matmul(W_row.transpose(-2, -1), grad_Y_col);
            // dX_col shape: [Batch, C_in*K_H*K_W, H_out*W_out]
            
            // col2im to accumulate gradients back to image dimensions [Batch, C_in, H_in, W_in]
            Tensor<real> grad_X = col2im(dX_col, W_, X_.shape(), stride_, padding_);
            X_.add_grad(grad_X);
        }
    }

    std::vector<Tensor<real>> get_inputs() const override {
        std::vector<Tensor<real>> inputs;
        if(X_.requires_grad()) inputs.push_back(X_);
        if(W_.requires_grad()) inputs.push_back(W_);
        if(bias_.numel() > 0 && bias_.requires_grad()) inputs.push_back(bias_);
        return inputs;
    }
};

}


