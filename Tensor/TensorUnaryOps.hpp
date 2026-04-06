#pragma once
namespace torch{
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
Tensor<real> Tensor<real>::contiguous() const{
    if(is_contiguous()) return *this;

    Tensor<real> output(shape_, requires_grad_);

    int ndim = static_cast<int>(shape_.size());
    if(ndim==0){
        output.data_[0] = data_[0];
        return output;
    }
    std::vector<int> curr_indices(ndim,0);
    int old_offset=0,new_offset=0;
    while(new_offset < numel_){
        output.data_[new_offset++] = data_[old_offset];
        for(int d = ndim-1;d>=0;--d){
            ++curr_indices[d];
            if(curr_indices[d] < shape_[d]){
                old_offset += stride_[d];
                break;
            } else {
                old_offset -= stride_[d] * shape_[d];
                curr_indices[d] = 0;
            }
        }
    }
    return output;
}
}