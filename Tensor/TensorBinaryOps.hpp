#pragma once
namespace torch{
template<typename real>
Tensor<real> Tensor<real>::matmal(const Tensor<real>& other){
    
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
Tensor<real> operator+(real A,const Tensor<real>& B){
    Tensor<real> output(B.shape_);
    for(int i=0;i<B.numel_;++i) output.data_[i] = A + B.data_[i];
    if(B.requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<AddBackwardScaler<real>> grad_fn = std::make_shared<AddBackwardScaler<real>>(B);
        output.set_grad_fn(grad_fn);
    }
    return output;
}

template<typename real>
Tensor<real> operator+ (const Tensor<real>& A,real B){
    Tensor<real> output(A.shape_);
    for(int i=0;i<A.numel_;++i) output.data_[i] = B + A.data_[i];
    if(A.requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<AddBackwardScaler<real>> grad_fn = std::make_shared<AddBackwardScaler<real>>(A);
        output.set_grad_fn(grad_fn);
    }
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
    if(A.requires_grad() || B.requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<AddBackward<real>> grad_fn = std::make_shared<AddBackward<real>>(A,B);
        output.set_grad_fn(grad_fn);
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
}