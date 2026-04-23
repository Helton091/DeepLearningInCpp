#ifndef TENSOR_HPP_
#define TENSOR_HPP_
#include<vector>
#include<memory>
#include<stdexcept>
#include<algorithm>
#include<iostream>
#include<string>
using std::ostream,std::string;
namespace torch{
class BroadCastingError : public std::invalid_argument{
public:
    BroadCastingError(const std::vector<int>& shape1,const std::vector<int>& shape2,int op,const string& str = "") : std::invalid_argument(build_message(shape1,shape2,op,str)){}
private:
    string build_message(const std::vector<int>& shape1,const std::vector<int>& shape2,int op,const string& str);
};

template<typename real>
class Tensor;

template<typename real>
class BackwardFunction;

template<typename real>
struct AutogradMeta;

template<typename real>                  
std::ostream& operator<<(std::ostream& os, const Tensor<real>& t);

template<typename real>
Tensor<real> operator+(const Tensor<real>& A,const Tensor<real>& B);

template<typename real>
Tensor<real> operator+ (const Tensor<real>& A,real B);

template<typename real>
Tensor<real> operator+ (real A,const Tensor<real>& B);

template<typename real>
Tensor<real> operator-(const Tensor<real>& A,const Tensor<real>& B);

template<typename real>
Tensor<real> operator- (const Tensor<real>& A,real B);

template<typename real>
Tensor<real> operator- (real A,const Tensor<real>& B);

template<typename real>
Tensor<real> operator*(const Tensor<real>& A,const Tensor<real>& B);

template<typename real>
Tensor<real> operator* (const Tensor<real>& A,real B);

template<typename real>
Tensor<real> operator* (real A,const Tensor<real>& B);

template<typename real>
Tensor<real> operator/(const Tensor<real>& A,const Tensor<real>& B);

template<typename real>
Tensor<real> operator/ (const Tensor<real>& A,real B);

template<typename real>
Tensor<real> operator/ (real A,const Tensor<real>& B);
template<typename real>
class Tensor{
private:
    std::shared_ptr<real []> data_ = nullptr; 
    std::vector<int> stride_; //how to fetch element in data_
    std::vector<int> shape_; 
    size_t numel_ = 0; //total element
    std::shared_ptr<AutogradMeta<real>> autograd_meta_ = nullptr;
    //Tensor/TensorUtilities.hpp
    void print_recursive(std::ostream& os, int dim_index, int current_offset) const;
    void init_metadata(); //shape_ -> stride_ + numel_
    Tensor(const std::vector<int>& shape,std::shared_ptr<real[]> shared_data,bool requires_grad);
    Tensor(const std::vector<int>& shape,std::vector<int>& stride,std::shared_ptr<real []> shared_data,bool requires_grad);
public:
    Tensor() = default; // REQUIRED for AutogradMeta default construction
    size_t numel() const noexcept{return numel_;}
    const real* data_ptr() const{return data_.get();}
    real* data_ptr(){return data_.get();}
    const std::vector<int>& shape()const noexcept{return shape_;}
    void fill_(real value);
    bool is_contiguous() const;
    const std::vector<int>& stride()const noexcept{return stride_;}
    bool requires_grad() const noexcept{ return autograd_meta_ && autograd_meta_->requires_grad;}
    void set_requires_grad(bool require);
    AutogradMeta<real>* get_autograd_meta() const{return autograd_meta_.get();}

    Tensor(const std::vector<int>& shape,bool requires_grad = false);
    Tensor(Tensor&& other) noexcept = default;
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    
    //Tensor/TensorBinaryOps.hpp
    friend ostream & operator<< <>(ostream & os,const Tensor<real>& t);
    friend Tensor<real> operator+ <>(const Tensor<real>& A,const Tensor<real>& B);
    friend Tensor<real> operator+ <>(const Tensor<real>& A,real B);
    friend Tensor<real> operator+ <>(real A,const Tensor<real>& B);
    friend Tensor<real> operator- <>(const Tensor<real>& A,const Tensor<real>& B);
    friend Tensor<real> operator- <>(const Tensor<real>& A,real B);
    friend Tensor<real> operator- <>(real A,const Tensor<real>& B);
    friend Tensor<real> operator* <>(const Tensor<real>& A,const Tensor<real>& B);
    friend Tensor<real> operator* <>(const Tensor<real>& A,real B);
    friend Tensor<real> operator* <>(real A,const Tensor<real>& B);
    friend Tensor<real> operator/ <>(const Tensor<real>& A,const Tensor<real>& B);
    friend Tensor<real> operator/ <>(const Tensor<real>& A,real B);
    friend Tensor<real> operator/ <>(real A,const Tensor<real>& B);
    Tensor<real> matmal(const Tensor<real>& other);
    //Tensor/TensorUnaryOps.hpp
    Tensor<real> reshape(std::vector<int> new_shape) const;
    Tensor<real> transpose(int dim0, int dim1) const;
    Tensor<real> permute(std::vector<int> dims) const;
    Tensor<real> contiguous() const; //return a new tensor with decreasing stride
    Tensor<real> squeeze(int dim) const;
    Tensor<real> squeeze() const;
    Tensor<real> unsqueeze(int dim) const;
    Tensor<real> sum(int dim,bool keep_dim=false) const;
    void mul_(real b){for(int i=0;i<numel_;++i) data_[i] = data_[i] * b;}
    void add_(real b){for(int i=0;i<numel_;++i) data_[i] = data_[i] + b;}
    void sub_(real b){for(int i=0;i<numel_;++i) data_[i] = data_[i] - b;}
    void div_(real b){for(int i=0;i<numel_;++i) data_[i] = data_[i] / b;}
    // Torch\TorchBackwardFunctions.hpp
    bool is_leaf() const;
    std::shared_ptr<BackwardFunction<real>> grad_fn() const;
    void add_grad(const Tensor<real>& g);
    Tensor<real> grad() const;
    void set_grad_fn(std::shared_ptr<BackwardFunction<real>> fn);
    void backward();

};
}

#include"Tensor/TensorUtilities.hpp"
#include"Tensor/TensorBinaryOps.hpp"
#include"Tensor/TensorUnaryOps.hpp"
#include"Torch/AutoGrad/TorchBackwardFunctions.hpp"
#include"Torch/AutoGrad/AutoGrad.hpp"
#endif