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

template<typename real = float>
Tensor<real> im2col(const Tensor<real>& X,const Tensor<real>& W,int stride,int padding = 0);
//X [B,C_in,H_in,W_in]

template<typename real = float>
Tensor<real> col2im(const Tensor<real>& X_col, const Tensor<real>& W, const std::vector<int>& X_shape, int stride, int padding = 0);
//X_col [B, C_in*K_H*K_W, H_out*W_out] -> returns [B, C_in, H_in, W_in]

}

namespace torch{
template<typename real>
Tensor<real> conv2d(const Tensor<real>& X,const Tensor<real>& W,const Tensor<real>& bias,int stride,int padding){
    const int X_ndim = X.shape().size();
    const int W_ndim = W.shape().size();
    if(X_ndim < 3 || W_ndim != 4) throw std::invalid_argument("conv2d: X must be at least 3D [..., C_in, H_in, W_in] and W must be 4D [C_out, C_in, K_H, K_W]");
    
    const int C_out = W.shape()[0];
    const int C_in_W = W.shape()[1];
    const int K_H = W.shape()[2];
    const int K_W = W.shape()[3];
    
    const int H_in = X.shape()[X_ndim - 2];
    const int W_in = X.shape()[X_ndim - 1];
    const int C_in_X = X.shape()[X_ndim - 3];
    
    if(C_in_X != C_in_W) throw std::invalid_argument("conv2d: Input channels must match Weight channels");
    
    // Calculate total batch size handling multidimensional batches [..., C, H, W]
    int Batch_size = 1;
    for(int i = 0; i < X_ndim - 3; ++i) Batch_size *= X.shape()[i];
    
    const int H_out = ((H_in + padding*2 - K_H) / stride) + 1;
    const int W_out = ((W_in + padding*2 - K_W) / stride) + 1;
    
    // 1. im2col: Transform input X into X_col
    // X_col shape: [Batch_size, C_in * K_H * K_W, H_out * W_out]
    Tensor<real> X_col = im2col(X, W, stride, padding);
    
    // 2. Reshape weight W into a 2D matrix W_row
    // W_row shape: [C_out, C_in * K_H * K_W]
    Tensor<real> W_row = W.reshape({C_out, C_in_X * K_H * K_W});
    
    // 3. Matrix Multiplication (GEMM)
    // We want: Y_col = W_row @ X_col
    // W_row: [C_out, K]
    // X_col: [Batch_size, K, N] -> we need batched matmul, so we treat W_row as [1, C_out, K] or let batched matmul broadcast it
    // Wait, our matmul supports broadcasting!
    Tensor<real> Y_col = W_row.matmal(X_col); 
    // Y_col shape: [Batch_size, C_out, H_out * W_out]
    
    // 4. Reshape back to image format
    // Y_out shape: [Batch_size, C_out, H_out, W_out]
    std::vector<int> out_shape = X.shape();
    out_shape[X_ndim - 3] = C_out;
    out_shape[X_ndim - 2] = H_out;
    out_shape[X_ndim - 1] = W_out;
    
    Tensor<real> Y_out = Y_col.reshape(out_shape);
    
    // 5. Add bias (if provided)
    if(bias.numel() > 0) {
        if(bias.shape().size() != 1 || bias.shape()[0] != C_out) {
            throw std::invalid_argument("conv2d: bias must be 1D with shape [C_out]");
        }
        // Bias needs to be broadcastable to [Batch_size, C_out, H_out, W_out]
        // We can reshape bias to [1, C_out, 1, 1] (or [C_out, 1, 1] if X is 3D)
        std::vector<int> bias_shape(X_ndim, 1);
        bias_shape[X_ndim - 3] = C_out;
        Tensor<real> bias_reshaped = bias.reshape(bias_shape);
        
        Y_out = Y_out + bias_reshaped;
    }
    
    if (is_grad_enabled && (X.requires_grad() || W.requires_grad() || (bias.numel() > 0 && bias.requires_grad()))) {
        Y_out.set_requires_grad(true);
        Y_out.set_grad_fn(std::make_shared<Conv2dBackward<real>>(X, W, bias, X_col, stride, padding));
    }
    
    return Y_out;
}
//X : [...,C_in,H_in,W_in]  W : [C_out,C_in,K_H,K_W]

}

namespace torch{

template<typename real>
Tensor<real> im2col(const Tensor<real>& X,const Tensor<real>& W,int stride,int padding){
    const int K_H = W.shape()[2];
    const int K_W = W.shape()[3];
    const int X_ndim = X.shape().size();
    const int H_in = X.shape()[X_ndim - 2];
    const int W_in = X.shape()[X_ndim - 1];
    const int C_in = X.shape()[X_ndim - 3];
    const int Batch_size = X.numel() / (H_in * W_in * C_in);
    
    const int H_out = ((H_in + padding*2 - K_H) / stride) + 1;
    const int W_out = ((W_in + padding*2 - K_W) / stride) + 1;
    
    // X_col shape: [Batch_size, C_in * K_H * K_W, H_out * W_out]
    Tensor<real> X_col({Batch_size, C_in * K_H * K_W, H_out * W_out}, false);
    
    // Ensure X is contiguous for safe flat memory access
    Tensor<real> X_contig = X.contiguous();
    const real* x_data = X_contig.data_ptr();
    real* col_data = X_col.data_ptr();
    
    const int channel_size = H_in * W_in;
    const int batch_size_x = C_in * channel_size;
    const int col_height = C_in * K_H * K_W;
    const int col_width = H_out * W_out;
    
    for (int b = 0; b < Batch_size; ++b) {
        for (int c = 0; c < C_in; ++c) {
            for (int kh = 0; kh < K_H; ++kh) {
                for (int kw = 0; kw < K_W; ++kw) {
                    // col_row is the row index in the output X_col matrix
                    int col_row = c * (K_H * K_W) + kh * K_W + kw;
                    
                    for (int ho = 0; ho < H_out; ++ho) {
                        for (int wo = 0; wo < W_out; ++wo) {
                            // col_col is the column index in the output X_col matrix
                            int col_col = ho * W_out + wo;
                            
                            // Map output spatial coordinate back to input spatial coordinate
                            int h_in_pos = ho * stride - padding + kh;
                            int w_in_pos = wo * stride - padding + kw;
                            
                            // Calculate 1D index for X_col
                            int col_idx = b * (col_height * col_width) + col_row * col_width + col_col;
                            
                            // Check if the mapped input coordinate is within bounds (padding handling)
                            if (h_in_pos >= 0 && h_in_pos < H_in && w_in_pos >= 0 && w_in_pos < W_in) {
                                int x_idx = b * batch_size_x + c * channel_size + h_in_pos * W_in + w_in_pos;
                                col_data[col_idx] = x_data[x_idx];
                            } else {
                                col_data[col_idx] = static_cast<real>(0.0); // Zero padding
                            }
                        }
                    }
                }
            }
        }
    }
    return X_col;
}

template<typename real>
Tensor<real> col2im(const Tensor<real>& X_col, const Tensor<real>& W, const std::vector<int>& X_shape, int stride, int padding) {
    const int K_H = W.shape()[2];
    const int K_W = W.shape()[3];
    const int X_ndim = X_shape.size();
    const int H_in = X_shape[X_ndim - 2];
    const int W_in = X_shape[X_ndim - 1];
    const int C_in = X_shape[X_ndim - 3];
    
    int Batch_size = 1;
    for(int i = 0; i < X_ndim - 3; ++i) Batch_size *= X_shape[i];

    const int H_out = ((H_in + padding*2 - K_H) / stride) + 1;
    const int W_out = ((W_in + padding*2 - K_W) / stride) + 1;

    Tensor<real> X(X_shape, false);
    X.fill_(static_cast<real>(0.0)); // Crucial: Initialize with zeros for scatter add
    
    Tensor<real> X_col_contig = X_col.contiguous();
    const real* col_data = X_col_contig.data_ptr();
    real* x_data = X.data_ptr();

    const int channel_size = H_in * W_in;
    const int batch_size_x = C_in * channel_size;
    const int col_height = C_in * K_H * K_W;
    const int col_width = H_out * W_out;

    for (int b = 0; b < Batch_size; ++b) {
        for (int c = 0; c < C_in; ++c) {
            for (int kh = 0; kh < K_H; ++kh) {
                for (int kw = 0; kw < K_W; ++kw) {
                    int col_row = c * (K_H * K_W) + kh * K_W + kw;
                    
                    for (int ho = 0; ho < H_out; ++ho) {
                        for (int wo = 0; wo < W_out; ++wo) {
                            int col_col = ho * W_out + wo;
                            
                            int h_in_pos = ho * stride - padding + kh;
                            int w_in_pos = wo * stride - padding + kw;
                            
                            int col_idx = b * (col_height * col_width) + col_row * col_width + col_col;
                            
                            // Scatter Add: accumulate gradients back to image pixels
                            if (h_in_pos >= 0 && h_in_pos < H_in && w_in_pos >= 0 && w_in_pos < W_in) {
                                int x_idx = b * batch_size_x + c * channel_size + h_in_pos * W_in + w_in_pos;
                                x_data[x_idx] += col_data[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return X;
}

template<typename real>
Tensor<real> matmul(const Tensor<real>& A,const Tensor<real>& B){
    int a_ndim = A.shape().size(),b_ndim = B.shape().size();
    if(a_ndim<=0 || b_ndim<=0) throw std::invalid_argument("for matmal, A or B can't be scaler");
    if(a_ndim==1 && b_ndim==1){
        Tensor<real> output = dot(A,B);
        if(is_grad_enabled && (A.requires_grad() || B.requires_grad())){
            output.set_requires_grad(true);
            std::shared_ptr<MatMulBackward<real>> grad_fn = std::make_shared<MatMulBackward<real>>(A,B);
            output.set_grad_fn(grad_fn);
        }
        return output;
    }
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
        if(is_grad_enabled && (A.requires_grad() || B.requires_grad())){
            output.set_requires_grad(true);
            std::shared_ptr<MatMulBackward<real>> grad_fn = std::make_shared<MatMulBackward<real>>(A,B);
            output.set_grad_fn(grad_fn);
        }
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
        if(is_grad_enabled && (A.requires_grad() || B.requires_grad())){
            output.set_requires_grad(true);
            std::shared_ptr<MatMulBackward<real>> grad_fn = std::make_shared<MatMulBackward<real>>(A,B);
            output.set_grad_fn(grad_fn);
        }
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