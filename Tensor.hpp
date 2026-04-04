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
    
    void fill_(real value);
};

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
Tensor<real> operator+(real A,const Tensor<real>& B){
    Tensor<real> output(B.shape_);
    for(int i=0;i<B.numel_;++i) output.data_[i] = A + B.data_[i];
    return output;
}

template<typename real>
Tensor<real> operator+ (const Tensor<real>& A,real B){
    Tensor<real> output(A.shape_);
    for(int i=0;i<A.numel_;++i) output.data_[i] = B + A.data_[i];
    return output;
}
template<typename real>
Tensor<real> operator+(const Tensor<real>& A,const Tensor<real>& B){
    
    int a_ndim = A.shape_.size();
    int b_ndim = B.shape_.size();
    int max_ndim = std::max(a_ndim,b_ndim);
    std::vector<int> output_shape;
    output_shape.resize(max_ndim);
    //step 1: check broadcasting and determine output_shape
    for(int i=0;i<max_ndim;++i){
        int a_idx = a_ndim - i - 1;
        int b_idx = b_ndim - i - 1;
        int a_dim = (a_idx >= 0) ? A.shape_[a_idx] : 1;
        int b_dim = (b_idx >= 0) ? B.shape_[b_idx] : 1;
        if(a_dim!=1 && b_dim!=1 && a_dim!=b_dim){
            throw BroadCastingError(A.shape_,B.shape_,0,"Tensor shape mismatch when broadcasting");
        }
        output_shape[max_ndim - i - 1] = std::max(a_dim,b_dim);
    }
    //step2: addition
    Tensor<real> output(output_shape);
    for(int i=0;i<output.numel_;++i){
        int res = i;
        std::vector<int> indices_out;
        indices_out.resize(max_ndim);
        for(int j=0;j<max_ndim;++j){
            if(output.stride_[j]==0) indices_out[j]=0;
            else{indices_out[j] = res / output.stride_[j];
            res = (res % output.stride_[j]);}
        }
        int i1=0,i2=0;
        for(int j=0;j<max_ndim;++j){
            int a_idx=0,b_idx=0;
            a_idx = j - (max_ndim - a_ndim);
            b_idx = j - (max_ndim - b_ndim);
            if(a_idx >= 0 && A.shape_[a_idx] == output.shape_[j]){
                i1 += indices_out[j] * A.stride_[a_idx];
            }
            if(b_idx >= 0 && B.shape_[b_idx] == output.shape_[j]){
                i2 += indices_out[j] * B.stride_[b_idx];
            }
        }
        output.data_[i] = A.data_[i1] + B.data_[i2];
    }
    return output;
}

template<typename real>
Tensor<real> operator-(real A, const Tensor<real>& B){
    Tensor<real> output(B.shape_);
    for(int i = 0; i < B.numel_; ++i) {
        output.data_[i] = A - B.data_[i];
    }
    return output;
}

template<typename real>
Tensor<real> operator-(const Tensor<real>& A, real B){
    Tensor<real> output(A.shape_);
    for(int i = 0; i < A.numel_; ++i) {
        output.data_[i] = A.data_[i] - B;
    }
    return output;
}

template<typename real>
Tensor<real> operator-(const Tensor<real>& A, const Tensor<real>& B){
    int a_ndim = A.shape_.size();
    int b_ndim = B.shape_.size();
    int max_ndim = std::max(a_ndim, b_ndim);
    std::vector<int> output_shape;
    output_shape.resize(max_ndim);
    
    // step 1
    for(int i = 0; i < max_ndim; ++i){
        int a_idx = a_ndim - i - 1;
        int b_idx = b_ndim - i - 1;
        int a_dim = (a_idx >= 0) ? A.shape_[a_idx] : 1;
        int b_dim = (b_idx >= 0) ? B.shape_[b_idx] : 1;
        
        if(a_dim != 1 && b_dim != 1 && a_dim != b_dim){
            // op = 1 represents subtracting
            throw BroadCastingError(A.shape_, B.shape_, 1, "Tensor shape mismatch when broadcasting");
        }
        output_shape[max_ndim - i - 1] = std::max(a_dim, b_dim);
    }
    
    // step 2
    Tensor<real> output(output_shape);
    for(int i = 0; i < output.numel_; ++i){
        int res = i;
        std::vector<int> indices_out;
        indices_out.resize(max_ndim);
        for(int j = 0; j < max_ndim; ++j){
            if(output.stride_[j] == 0) {
                indices_out[j] = 0;
            } else {
                indices_out[j] = res / output.stride_[j];
                res = (res % output.stride_[j]);
            }
        }
        
        int i1 = 0, i2 = 0;
        for(int j = 0; j < max_ndim; ++j){
            int a_idx = 0, b_idx = 0;
            a_idx = j - (max_ndim - a_ndim);
            b_idx = j - (max_ndim - b_ndim);
            if(a_idx >= 0 && A.shape_[a_idx] == output.shape_[j]){
                i1 += indices_out[j] * A.stride_[a_idx];
            }
            if(b_idx >= 0 && B.shape_[b_idx] == output.shape_[j]){
                i2 += indices_out[j] * B.stride_[b_idx];
            }
        }
        output.data_[i] = A.data_[i1] - B.data_[i2];
    }
    return output;
}

template<typename real>
Tensor<real> operator*(real A, const Tensor<real>& B){
    Tensor<real> output(B.shape_);
    for(int i = 0; i < B.numel_; ++i) {
        output.data_[i] = A * B.data_[i];
    }
    return output;
}

template<typename real>
Tensor<real> operator*(const Tensor<real>& A, real B){
    Tensor<real> output(A.shape_);
    for(int i = 0; i < A.numel_; ++i) {
        output.data_[i] = A.data_[i] * B;
    }
    return output;
}

template<typename real>
Tensor<real> operator*(const Tensor<real>& A, const Tensor<real>& B){
    int a_ndim = A.shape_.size();
    int b_ndim = B.shape_.size();
    int max_ndim = std::max(a_ndim, b_ndim);
    std::vector<int> output_shape;
    output_shape.resize(max_ndim);
    
    // step 1
    for(int i = 0; i < max_ndim; ++i){
        int a_idx = a_ndim - i - 1;
        int b_idx = b_ndim - i - 1;
        int a_dim = (a_idx >= 0) ? A.shape_[a_idx] : 1;
        int b_dim = (b_idx >= 0) ? B.shape_[b_idx] : 1;
        
        if(a_dim != 1 && b_dim != 1 && a_dim != b_dim){
            // op = 2 represents element-wise-multiplying
            throw BroadCastingError(A.shape_, B.shape_, 2, "Tensor shape mismatch when broadcasting");
        }
        output_shape[max_ndim - i - 1] = std::max(a_dim, b_dim);
    }
    
    // step 2
    Tensor<real> output(output_shape);
    for(int i = 0; i < output.numel_; ++i){
        int res = i;
        std::vector<int> indices_out;
        indices_out.resize(max_ndim);
        for(int j = 0; j < max_ndim; ++j){
            if(output.stride_[j] == 0) {
                indices_out[j] = 0;
            } else {
                indices_out[j] = res / output.stride_[j];
                res = (res % output.stride_[j]);
            }
        }
        
        int i1 = 0, i2 = 0;
        for(int j = 0; j < max_ndim; ++j){
            int a_idx = 0, b_idx = 0;
            a_idx = j - (max_ndim - a_ndim);
            b_idx = j - (max_ndim - b_ndim);
            if(a_idx >= 0 && A.shape_[a_idx] == output.shape_[j]){
                i1 += indices_out[j] * A.stride_[a_idx];
            }
            if(b_idx >= 0 && B.shape_[b_idx] == output.shape_[j]){
                i2 += indices_out[j] * B.stride_[b_idx];
            }
        }
        output.data_[i] = A.data_[i1] * B.data_[i2];
    }
    return output;
}

template<typename real>
Tensor<real> operator/(real A, const Tensor<real>& B){
    Tensor<real> output(B.shape_);
    for(int i = 0; i < B.numel_; ++i) {
        if(B.data_[i] == static_cast<real>(0)){
            throw std::invalid_argument("Division by zero encountered in tensor elements");
        }
        output.data_[i] = A / B.data_[i];
    }
    return output;
}

template<typename real>
Tensor<real> operator/(const Tensor<real>& A, real B){
    if(B == static_cast<real>(0)){
        throw std::invalid_argument("Division by zero encountered with scalar");
    }
    Tensor<real> output(A.shape_);
    for(int i = 0; i < A.numel_; ++i) {
        output.data_[i] = A.data_[i] / B;
    }
    return output;
}

template<typename real>
Tensor<real> operator/(const Tensor<real>& A, const Tensor<real>& B){
    int a_ndim = A.shape_.size();
    int b_ndim = B.shape_.size();
    int max_ndim = std::max(a_ndim, b_ndim);
    std::vector<int> output_shape;
    output_shape.resize(max_ndim);
    
    // step 1
    for(int i = 0; i < max_ndim; ++i){
        int a_idx = a_ndim - i - 1;
        int b_idx = b_ndim - i - 1;
        int a_dim = (a_idx >= 0) ? A.shape_[a_idx] : 1;
        int b_dim = (b_idx >= 0) ? B.shape_[b_idx] : 1;
        
        if(a_dim != 1 && b_dim != 1 && a_dim != b_dim){
            // op = 3 represents element-wise-dividing
            throw BroadCastingError(A.shape_, B.shape_, 3, "Tensor shape mismatch when broadcasting");
        }
        output_shape[max_ndim - i - 1] = std::max(a_dim, b_dim);
    }
    
    // step 2
    Tensor<real> output(output_shape);
    for(int i = 0; i < output.numel_; ++i){
        int res = i;
        std::vector<int> indices_out;
        indices_out.resize(max_ndim);
        for(int j = 0; j < max_ndim; ++j){
            if(output.stride_[j] == 0) {
                indices_out[j] = 0;
            } else {
                indices_out[j] = res / output.stride_[j];
                res = (res % output.stride_[j]);
            }
        }
        
        int i1 = 0, i2 = 0;
        for(int j = 0; j < max_ndim; ++j){
            int a_idx = 0, b_idx = 0;
            a_idx = j - (max_ndim - a_ndim);
            b_idx = j - (max_ndim - b_ndim);
            if(a_idx >= 0 && A.shape_[a_idx] == output.shape_[j]){
                i1 += indices_out[j] * A.stride_[a_idx];
            }
            if(b_idx >= 0 && B.shape_[b_idx] == output.shape_[j]){
                i2 += indices_out[j] * B.stride_[b_idx];
            }
        }
        
        if(B.data_[i2] == static_cast<real>(0)){
            throw std::invalid_argument("Division by zero encountered in tensor elements");
        }
        output.data_[i] = A.data_[i1] / B.data_[i2];
    }
    return output;
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
#endif