#pragma once
namespace torch{

template<typename real>
Tensor<real> Tensor<real>::sum() const{
    Tensor<real> contiguous_tensor = contiguous();
    Tensor<real> output({},requires_grad());
    real* output_data_ptr = output.data_.get();
    real* curr_data_ptr = contiguous_tensor.data_.get();
    *output_data_ptr = static_cast<real>(0.0);
    for(int i=0;i<numel_;++i){
        *output_data_ptr += curr_data_ptr[i];
    }
    if(requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<SumAllBackward<real>> grad_fn = std::make_shared<SumAllBackward<real>>(*this,this->shape());
        output.set_grad_fn(grad_fn);
    }
    return output;
}
template<typename real>
Tensor<real> Tensor<real>::sum(int dim,bool keep_dim) const{
    int ndim = shape_.size();
    if (ndim == 0) {
        if (dim != 0 && dim != -1) {
            throw std::out_of_range("Dimension out of range (expected to be in range of [-1, 0] ...)");
        }
        Tensor<real> output({}, requires_grad());
        output.data_ptr()[0] = data_[0];
        return output;
    }
    if(dim < 0) dim += ndim;
    if(dim < 0 || dim >= ndim) throw std::out_of_range("index out of bounds when calling Tensor::sum");
    if(keep_dim){
        std::vector<int> new_shape(ndim);
        for(int i=0;i<ndim;++i){
            if(i==dim) new_shape[i] = 1;
            else new_shape[i] = shape_[i];
        }
        Tensor<real> output(new_shape,requires_grad());
        output.fill_(static_cast<real>(0.0));
        std::vector<int> old_shape_indices(ndim,0);
        for(int i=0;i<numel_;++i){
            int output_offset = 0;
            int curr_offset = 0;
            for(int j=0;j<ndim;++j){
                if(j!=dim) output_offset += old_shape_indices[j] * output.stride_[j];
                curr_offset += old_shape_indices[j] * stride_[j];
            }
            output.data_[output_offset] += data_[curr_offset];
            for(int j=ndim-1;j>=0;--j){
                ++old_shape_indices[j];
                if(old_shape_indices[j] < shape_[j]) break;
                else old_shape_indices[j] = 0;
            }
        }
        if(is_grad_enabled && requires_grad()){
            output.set_requires_grad(true);
            std::shared_ptr<SumBackward<real>> grad_fn = std::make_shared<SumBackward<real>>(*this, dim, keep_dim, shape_);
            output.set_grad_fn(grad_fn);
        }
        return output;
    } else {
        std::vector<int> new_shape(ndim - 1);
        int temp_idx = 0;
        for(int i=0;i<ndim;++i){
            if(i!=dim) new_shape[temp_idx++] = shape_[i];
        }
        Tensor<real> output(new_shape,requires_grad());
        output.fill_(static_cast<real>(0.0));
        std::vector<int> old_shape_indices(ndim,0);
        for(int i=0;i<numel_;++i){
            int output_offset = 0;
            int curr_offset = 0;
            for(int j=0;j<ndim;++j){
                if(j<dim) output_offset += old_shape_indices[j] * output.stride_[j];
                if(j>dim) output_offset += old_shape_indices[j] * output.stride_[j-1];
                curr_offset += old_shape_indices[j] * stride_[j];
            }
            output.data_[output_offset] += data_[curr_offset];
            for(int j=ndim-1;j>=0;--j){
                ++old_shape_indices[j];
                if(old_shape_indices[j] < shape_[j]) break;
                else old_shape_indices[j] = 0;
            }
        }
        if(is_grad_enabled && requires_grad()){
            output.set_requires_grad(true);
            std::shared_ptr<SumBackward<real>> grad_fn = std::make_shared<SumBackward<real>>(*this, dim, keep_dim, shape_);
            output.set_grad_fn(grad_fn);
        }
        return output;
    }
}

template<typename real> 
Tensor<real> Tensor<real>::unsqueeze(int dim) const { 
    int ndim = static_cast<int>(shape_.size()); 
    
    int real_dim = dim < 0 ? dim + ndim + 1 : dim; 
    
    if (real_dim < 0 || real_dim > ndim) {
        throw std::out_of_range("index out of range when unsqueezing"); 
    }

    std::vector<int> new_shape = shape_; 
    std::vector<int> new_stride = stride_; 
    
    
    new_shape.insert(new_shape.begin() + real_dim, 1); 

   
    int new_stride_val = 1; 
    if (real_dim < ndim) {
        new_stride_val = shape_[real_dim] * stride_[real_dim];
    }
    
    new_stride.insert(new_stride.begin() + real_dim, new_stride_val); 
    
    Tensor<real> output(new_shape, new_stride, data_, requires_grad());
    if(is_grad_enabled && requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<UnSqueezeBackward<real>> grad_fn = std::make_shared<UnSqueezeBackward<real>>(*this,dim);
        output.set_grad_fn(grad_fn);
    }
    return output;
} 

template<typename real>
Tensor<real> Tensor<real>::squeeze() const{
    int ndim_ori = static_cast<int>(shape_.size());
    int ndim_curr = ndim_ori;
    bool requires_squeeze = false;
    for(int i=0;i<ndim_ori;++i){
        if(shape_[i]==1){
            requires_squeeze = true;
            --ndim_curr;
        }
    }
    if(!requires_squeeze) return *this;
    std::vector<int> new_shape(ndim_curr),new_stride(ndim_curr);
    int j=0;
    for(int i=0;i<ndim_curr;++i,++j){
        while(shape_[j]==1) ++j;
        new_shape[i] = shape_[j];
        new_stride[i] = stride_[j];
    }
    Tensor<real> output(new_shape, new_stride, data_, requires_grad());
    if(is_grad_enabled && requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<SqueezeAllBackward<real>> grad_fn = std::make_shared<SqueezeAllBackward<real>>(*this,shape());
        output.set_grad_fn(grad_fn);
    }
    return output;
}


template<typename real>
Tensor<real> Tensor<real>::squeeze(int dim) const{
    int ndim = static_cast<int>(shape_.size());
    if(dim < 0) dim += ndim;
    if(dim < 0 || dim>=ndim) throw std::out_of_range("index out of range when squeezing");
    if(shape_[dim] != 1) return *this;

    std::vector<int> new_shape = shape_;
    std::vector<int> new_stride = stride_;

    new_shape.erase(new_shape.begin() + dim);
    new_stride.erase(new_stride.begin()+ dim);
    Tensor<real> output(new_shape, new_stride, data_, requires_grad());
    if(is_grad_enabled && requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<SqueezeBackward<real>> grad_fn = std::make_shared<SqueezeBackward<real>>(*this,dim);
        output.set_grad_fn(grad_fn);
    }
    return output;
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

    Tensor<real> output(new_shape, data_, requires_grad()); 
    if(is_grad_enabled && requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<ReshapeBackward<real>> grad_fn = std::make_shared<ReshapeBackward<real>>(*this,shape());
        output.set_grad_fn(grad_fn);
    }
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
    Tensor<real> output(new_shape, new_stride, data_, requires_grad());
    if(is_grad_enabled && requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<PermuteBackward<real>> grad_fn = std::make_shared<PermuteBackward<real>>(*this,shape());
        output.set_grad_fn(grad_fn);
    }
    return output;
}

template<typename real>
Tensor<real> Tensor<real>::transpose(int dim0, int dim1) const{
    int ndim = static_cast<int>(shape_.size());
    if(dim0 < 0) dim0 += ndim;
    if(dim1 < 0) dim1 += ndim;
    if(dim0<0 || dim1<0 || dim0>=ndim || dim1>=ndim){
        throw std::invalid_argument("dimension out of bound when transposing");
    }
    if(dim0==dim1){ 
        return *this;
    }

    std::vector<int> new_shape = shape_;
    std::vector<int> new_stride = stride_;
    
    std::swap(new_shape[dim0],new_shape[dim1]);
    std::swap(new_stride[dim0],new_stride[dim1]);
    Tensor<real> output(new_shape, new_stride, data_, requires_grad());
    if(is_grad_enabled && requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<TransposeBackward<real>> grad_fn = std::make_shared<TransposeBackward<real>>(*this,dim0,dim1);
        output.set_grad_fn(grad_fn);
    }
    return output;
}

template<typename real>
Tensor<real> Tensor<real>::contiguous() const{
    if(is_contiguous()) return *this;

    Tensor<real> output(shape_, requires_grad());

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
                old_offset -= stride_[d] * (shape_[d]-1);
                curr_indices[d] = 0;
            }
        }
    }
    if(is_grad_enabled &&  requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<ContiguousBackward<real>> grad_fn = std::make_shared<ContiguousBackward<real>>(*this);
        output.set_grad_fn(grad_fn);
    }
    return output;
}

template<typename real>
Tensor<real> Tensor<real>::expand(const std::vector<int>& target_shape) const{
    int ndim = shape_.size();
    int target_ndim = target_shape.size();
    if (target_ndim < ndim) {
        throw std::invalid_argument("target_shape must have same or more dimensions than tensor");
    }    
    std::vector<int> target_stride(target_ndim,0);
    for(int i=0;i<ndim;++i){
        int real_idx_origin = ndim - i - 1;
        int real_idx_target = target_ndim - i - 1;
        if(target_shape[real_idx_target] == shape_[real_idx_origin]){
            target_stride[real_idx_target] = stride_[real_idx_origin];
        } 
        else if(shape_[real_idx_origin] != 1) {
            throw std::runtime_error("inable to broadcast when expanding");
        }
    }
    Tensor<real> output(target_shape,target_stride,data_,requires_grad());
    if(is_grad_enabled && requires_grad()){
        output.set_requires_grad(true);
        std::shared_ptr<ExpandBackward<real>> grad_fn = std::make_shared<ExpandBackward<real>>(*this);
        output.set_grad_fn(grad_fn);
    }
    return output;

}
}