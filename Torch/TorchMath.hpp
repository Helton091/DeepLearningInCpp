#pragma once
#include"../Tensor.hpp"
namespace torch{


template<typename real = float>
Tensor<real> dot(const Tensor<real>& A,const Tensor<real>& B);

template<typename real = float>
Tensor<real> matmul(const Tensor<real>& A,const Tensor<real>& B);

template<typename real = float>
void matmul_2d_core(const real* a_ptr,const real* b_ptr,real* c_ptr,
                    int M,int K,int N,
                    int stride_a_m,int stride_a_k,
                    int stride_b_k,int stride_b_n,
                    int stride_c_m,int stride_c_n);
}

namespace torch{
template<typename real>
Tensor<real> matmul(const Tensor<real>& A,const Tensor<real>& B){
    int a_ndim = A.shape().size(),b_ndim = B.shape().size();
    if(a_ndim<=0 || b_ndim<=0) throw std::invalid_argument("for matmal, A or B can't be scaler");
    if(a_ndim==1 && b_ndim==1) return dot(A,B);
    bool is_a_1d = (a_ndim == 1);
    bool is_b_1d = (b_ndim == 1);
    Tensor<real> A_temp = (is_a_1d) ? A.unsqueeze(0) : A;
    Tensor<real> B_temp = (is_b_1d) ? B.unsqueeze(-1) : B;
    if(A_temp.shape().size()==2 && B_temp.shape().size()==2){
        //normal 2d matmul 2d
        if(A_temp.shape()[1] != B_temp.shape()[0]) throw std::invalid_argument("for matrix multiplication, the second dimension of first matrix must be equal to the second matrix");
        Tensor<real> output({A_temp.shape()[0],B_temp.shape()[1]});
        output.fill_(static_cast<real>(0.0));
        matmul_2d_core(A_temp.data_ptr(),B_temp.data_ptr(),output.data_ptr(),A_temp.shape()[0],A_temp.shape()[1],B_temp.shape()[1],A_temp.stride()[0],A_temp.stride()[1],B_temp.stride()[0],B_temp.stride()[1],output.stride()[0],output.stride()[1]);
        if(is_a_1d) output = output.squeeze(0);
        if(is_b_1d) output = output.squeeze(-1);
        return output;
    } else {
        //batched matrix multiplication
        int a_temp_ndim = A_temp.shape().size();
        int b_temp_ndim = B_temp.shape().size();
        if(A_temp.shape()[a_temp_ndim-1] != B_temp.shape()[b_temp_ndim-2]) throw std::invalid_argument("matmul: inner dimensions must match in batched matmul");
        int max_ndim = std::max(a_temp_ndim,b_temp_ndim);
        std::vector<int> output_shape;
        output_shape.resize(max_ndim);
        int total_batch_num = 1;
        for(int i=0;i<max_ndim-2;++i){
            int a_idx = a_temp_ndim - i - 3;
            int b_idx = b_temp_ndim - i - 3;
            int a_dim = (a_idx < 0) ? 1 : A_temp.shape()[a_idx];
            int b_dim = (b_idx < 0) ? 1 : B_temp.shape()[b_idx];
            if(a_dim!=1 && b_dim!=1 && a_dim!=b_dim) throw std::invalid_argument("broadcasting failed when operating batched matrix multiplication");
            output_shape[max_ndim-3-i] = std::max(a_dim,b_dim);
            total_batch_num *= std::max(a_dim,b_dim);
        }
        output_shape[max_ndim-2] = A_temp.shape()[a_temp_ndim-2];
        output_shape[max_ndim-1] = B_temp.shape()[b_temp_ndim-1];
        
        Tensor<real> output(output_shape,(A_temp.requires_grad() || B_temp.requires_grad()));
        output.fill_(static_cast<real>(0.0));

        real* a_temp_ptr = A_temp.data_ptr();
        real* b_temp_ptr = B_temp.data_ptr();
        real* output_ptr = output.data_ptr();
        int M = output_shape[max_ndim - 2];
        int N = output_shape[max_ndim - 1];
        int K = A_temp.shape()[a_temp_ndim - 1];

        std::vector<int> batch_indices(max_ndim-2,0);
        for(int i=0;i<total_batch_num;++i){
            int a_temp_offset = 0,b_temp_offset = 0;
            int curr_batch_idx = i;
            
            for(int j=max_ndim-3;j>=0;--j){
                int a_idx = a_temp_ndim - j - 3;
                int b_idx = b_temp_ndim - j - 3;
                int a_dim = (a_idx < 0) ? 1 : A_temp.shape()[a_idx];
                int b_dim = (b_idx < 0) ? 1 : B_temp.shape()[b_idx];
                if(a_dim != 1) a_temp_offset += batch_indices[j] * A_temp.stride()[a_idx];
                if(b_dim != 1) b_temp_offset += batch_indices[j] * B_temp.stride()[b_idx];
            }
            matmul_2d_core(a_temp_ptr + a_temp_offset,b_temp_ptr + b_temp_offset,output_ptr + i*output_shape[max_ndim-1]*output_shape[max_ndim-2],
                            M,K,N,A_temp.stride()[a_temp_ndim-2],A_temp.stride()[a_temp_ndim-1],
                            B_temp.stride()[b_temp_ndim-2],B_temp.stride()[b_temp_ndim-1],
                            output.stride()[max_ndim-2],output.stride()[max_ndim-1]);
            for(int j=max_ndim-3;j>=0;--j){
                ++batch_indices[j];
                if(batch_indices[j] < output_shape[j]) break;
                else batch_indices[j] = 0;
            }

        }
        if(is_a_1d) output = output.squeeze(max_ndim - 2);
        if(is_b_1d) output = output.squeeze(max_ndim - 1);
        return output;
    }
}

template<typename real>
Tensor<real> dot(const Tensor<real>& A,const Tensor<real>& B){
    if(A.shape().size()!=1 || B.shape().size()!=1) throw std::invalid_argument("for dot product, their ndim must be 1");
    int a_length = A.shape()[0],b_length = B.shape()[0];
    if(a_length != b_length) throw std::invalid_argument("for dot product, the length of two tensor must be equal to each other");
    Tensor<real> output({},(A.requires_grad() || B.requires_grad()));
    real* output_ptr = output.data_ptr();
    const real* a_ptr = A.data_ptr();
    const real* b_ptr = B.data_ptr();
    int a_stride = A.stride()[0];
    int b_stride = B.stride()[0];
    real output_result = static_cast<real>(0.0);
    for(int i=0;i<a_length;++i) output_result += a_ptr[i*a_stride] * b_ptr[i*b_stride];
    output_ptr[0] = output_result;
    return output;
}

template<typename real>
void matmul_2d_core(const real* a_ptr,const real* b_ptr,real* c_ptr,
                    int M,int K,int N,
                    int stride_a_m,int stride_a_k,
                    int stride_b_k,int stride_b_n,
                    int stride_c_m,int stride_c_n){
    //C must be all zero before calculation
    for(int i=0;i<M;++i){
        for(int k=0;k<K;++k){
            real a_val = a_ptr[i*stride_a_m + k*stride_a_k];
            for(int j=0;j<N;++j){
                c_ptr[i*stride_c_m + j*stride_c_n] += a_val * b_ptr[k*stride_b_k + j*stride_b_n];
            }
        }
    }
    
}

}