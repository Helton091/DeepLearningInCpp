#pragma once
namespace torch{
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
Tensor<real>::Tensor(const std::vector<int>& shape,std::vector<int>& stride,std::shared_ptr<real []> shared_data,bool requires_grad){
    shape_ = shape;
    stride_ = stride;
    data_ = shared_data;
    requires_grad_ = requires_grad;
    numel_ = 1;
    for(int s : shape_) {
        numel_ *= s;
    }
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
}