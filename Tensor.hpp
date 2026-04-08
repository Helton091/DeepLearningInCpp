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
    std::shared_ptr<real []> data_; 
    std::vector<int> stride_; //how to fetch element in data_
    std::vector<int> shape_; 
    size_t numel_; //total element
    bool requires_grad_;
    //Tensor/TensorUtilities.hpp
    void print_recursive(std::ostream& os, int dim_index, int current_offset) const;
    void init_metadata(); //shape_ -> stride_ + numel_
    Tensor(const std::vector<int>& shape,std::shared_ptr<real[]> shared_data,bool requires_grad);
    Tensor(const std::vector<int>& shape,std::vector<int>& stride,std::shared_ptr<real []> shared_data,bool requires_grad);
public:
    size_t numel() const noexcept{return numel_;}
    const real* data_ptr() const{return data_.get();}
    real* data_ptr(){return data_.get();}
    const std::vector<int>& shape()const noexcept{return shape_;}
    void fill_(real value);
    bool is_contiguous() const;
    
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
    //Tensor/TensorUnaryOps.hpp
    Tensor<real> reshape(std::vector<int> new_shape) const;
    Tensor<real> transpose(int dim0, int dim1) const;
    Tensor<real> permute(std::vector<int> dims) const;
    Tensor<real> contiguous() const; //return a new tensor with decreasing stride
    Tensor<real> squeeze(int dim) const;
    Tensor<real> squeeze() const;
    Tensor<real> unsqueeze(int dim) const;
};
}

#include"Tensor/TensorUtilities.hpp"
#include"Tensor/TensorBinaryOps.hpp"
#include"Tensor/TensorUnaryOps.hpp"
#endif