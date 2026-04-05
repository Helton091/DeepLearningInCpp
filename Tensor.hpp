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
    void print_recursive(std::ostream& os, int dim_index, int current_offset) const;
    void init_metadata(); //shape_ -> stride_ + numel_
    Tensor(const std::vector<int>& shape,std::shared_ptr<real[]> shared_data,bool requires_grad);
    Tensor(const std::vector<int>& shape,std::vector<int>& stride,std::shared_ptr<real []> shared_data,bool requires_grad);
public:
    size_t numel() const noexcept{return numel_;}
    const real* data_ptr() const{return data_.get();}
    real* data_ptr(){return data_.get();}
    const std::vector<int>& shape()const noexcept{return shape_;}

    Tensor(const std::vector<int>& shape,bool requires_grad = false);
    Tensor(Tensor&& other) noexcept = default;
    
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
    
    Tensor<real> reshape(std::vector<int> new_shape) const;
    Tensor<real> transpose(int dim0, int dim1) const;
    Tensor<real> permute(std::vector<int> dims) const;
    Tensor<real> contiguous() const; //return a new tensor with decreasing stride
    bool is_contiguous() const;
    void fill_(real value);
};
template<typename real>
bool Tensor<real>::is_contiguous() const{
    int ndim = static_cast<int>(shape_.size());
    if(ndim==0) return true;
    int expected_stride = 1;
    for(int i=ndim-1;i>=0;--i){
        if(shape_[i] != 1 && stride_[i] != expected_stride) return false;
        //shape = 1 don't affect contiguous. this is a mismatch between theory and practice
        expected_stride *= shape_[i];
    }
    return true;
}

template<typename real>
Tensor<real> Tensor<real>::permute(std::vector<int> dims) const{
    if(dims.size() != shape_.size()){
        throw std::invalid_argument("dimension error when permuting vector");
    }
    int ndim = static_cast<int>(dims.size());
    std::vector<bool> seen(ndim,false);
    bool is_no_op = true;
    for(int i=0;i<ndim;++i){
        if(dims[i] < 0) dims[i] += ndim;
        if(dims[i] < 0 || dims[i] >= ndim) throw std::invalid_argument("dimension error when permuting vector");
        if(seen[dims[i]]) throw std::invalid_argument("dimension error when permuting vector");
        if(dims[i] != i) is_no_op=false;
        seen[dims[i]] = true;
    }
    if(is_no_op) return *this;
    std::vector<int> new_shape(ndim);
    std::vector<int> new_stride(ndim);
    for(int i=0;i<ndim;++i){
        new_shape[i] = shape_[dims[i]];
        new_stride[i] = stride_[dims[i]];
    }
    return Tensor<real>(std::move(new_shape),std::move(new_stride),data_,requires_grad_);
}
template<typename real>
Tensor<real>::Tensor(const std::vector<int>& shape,std::vector<int>& stride,std::shared_ptr<real []> shared_data,bool requires_grad){
    shape_ = shape;
    stride_ = stride;
    data_ = shared_data;
    requires_grad_ = requires_grad;
}

template<typename real>
Tensor<real> Tensor<real>::transpose(int dim0, int dim1) const{
    int ndim = static_cast<int>(shape_.size());
    if(dim0 < 0) dim0 += ndim;
    if(dim1 < 0) dim1 += ndim;
    if(dim0<0 || dim1<0 || dim0>=ndim || dim1>=ndim){
        throw std::invalid_argument("dimension out of bound when transposing");
    }
    if(dim0==dim1) return *this;

    std::vector<int> new_shape = shape_;
    std::vector<int> new_stride = stride_;
    
    std::swap(new_shape[dim0],new_shape[dim1]);
    std::swap(new_stride[dim0],new_stride[dim1]);
    
    return Tensor<real>(std::move(new_shape), std::move(new_stride), data_, requires_grad_);
}

template<typename real> 
Tensor<real> Tensor<real>::reshape(std::vector<int> new_shape) const {
    int minus_one_index = -1; 
    int curr_numel = 1; 

    for(int i = 0; i < new_shape.size(); ++i){ 
        if(new_shape[i] == -1){
            if(minus_one_index != -1) {
                throw std::invalid_argument("there is at most one -1 in the shape when reshaping"); 
            }
            minus_one_index = i; 
        } 
        else if (new_shape[i] <= 0) {
             throw std::invalid_argument("shape dimensions must be positive integers");
        }
        else {
            curr_numel *= new_shape[i]; 
        }
    } 

    if(minus_one_index != -1){ 
        
        if (numel_ % curr_numel != 0) {
            throw std::invalid_argument("unable to reshape due to numel mismatch (indivisible)");
        }
        new_shape[minus_one_index] = numel_ / curr_numel; 
        curr_numel *= new_shape[minus_one_index]; 
    } 

    if(curr_numel != numel_) {
        throw std::invalid_argument("unable to reshape due to numel mismatch"); 
    }

    Tensor<real> output(std::move(new_shape), data_, requires_grad_); 
    return output; 
} 
template<typename real>
Tensor<real>::Tensor(const std::vector<int>& shape,std::shared_ptr<real[]> shared_data,bool requires_grad){
    shape_ = shape;
    data_ = shared_data;
    init_metadata();
    requires_grad_ = requires_grad;
}

string BroadCastingError::build_message(const std::vector<int>& shape1,const std::vector<int>& shape2,int op,const string& str){
    string msg = "shape mismatch when ";
    switch(op){
    case 0:
        msg += "adding";
        break;
    case 1:
        msg += "subtracting";
        break;
    case 2:
        msg += "element-wise-multiplying";
        break;
    case 3:
        msg += "element-wise-dividing";
        break;
    }
    msg += " tensors with shape [";
    for(int dim : shape1){msg += std::to_string(dim);msg+=",";}
    msg += "] and [";
    for(int dim : shape2){msg += std::to_string(dim);msg+=',';}
    msg += "]";
    msg = msg + "\n" + str;
    return msg;
}
template<typename real>
void Tensor<real>::fill_(real value){
    std::fill_n(data_.get(), numel_, value);
}


template<typename real>
ostream & operator<< (ostream & os,const Tensor<real>& t){
    os << "tensor(";
    
    t.print_recursive(os, 0, 0);
    
    os << ", shape=[";
    for (size_t i = 0; i < t.shape_.size(); ++i) {
        os << t.shape_[i] << (i == t.shape_.size() - 1 ? "" : ", ");
    }
    os << "])";
    return os;
}

template<typename real>
void Tensor<real>::print_recursive(std::ostream& os, int dim_index, int current_offset) const {
    
    if (shape_.empty()) {
        os << data_[0];
        return;
    }

    
    if (dim_index == shape_.size() - 1) {
        os << '[';
        for (int i = 0; i < shape_[dim_index]; ++i) {
            os << data_[current_offset + i * stride_[dim_index]];
            
            if (i < shape_[dim_index] - 1) os << ", "; 
        }
        os << ']';
        return;
    }

    
    os << '[';
    for (int i = 0; i < shape_[dim_index]; ++i) {
        if (i > 0) {
            os << ',';
            
            int newlines = shape_.size() - dim_index - 1;
            for (int n = 0; n < newlines; ++n) os << '\n';
            
            
            int spaces_to_align = 7 + dim_index + 1;
            os << std::string(spaces_to_align, ' ');
        }
        
        int next_offset = current_offset + i * stride_[dim_index];
        print_recursive(os, dim_index + 1, next_offset);
    }
    os << ']';
}
template<typename real>
void Tensor<real>::init_metadata(){
    int dimension = shape_.size();

    if(dimension==0){
        numel_ = 1;
        stride_ = {};
        return ;
    }
    stride_.resize(dimension);
    numel_ = 1;
    for(int i=dimension-1;i>=0;--i){
        stride_[i] = numel_;
        numel_ *= shape_[i];
    }

}

template<typename real>
Tensor<real>::Tensor(const std::vector<int> & shape,bool requires_grad){
    shape_ = shape;
    requires_grad_ = requires_grad;
    init_metadata();
    data_ = std::shared_ptr<real []>(new real[numel_]);

}
}

#include"Tensor/TensorBinaryOps.hpp"
#include"Tensor/TensorUnaryOps.hpp"
#endif