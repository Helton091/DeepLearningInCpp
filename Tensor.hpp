#ifndef TENSOR_HPP_
#define TENSOR_HPP_
#include<vector>
#include<memory>
#include<stdexcept>
#include<algorithm>
#include<iostream>
using std::ostream;
template<typename real>
class Tensor;

template<typename real>                  
std::ostream& operator<<(std::ostream& os, const Tensor<real>& t);

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
    Tensor(const std::vector<int>& shape,bool requires_grad = false);
    friend ostream & operator<< <> (ostream & os,const Tensor<real>& t);
};
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
#endif