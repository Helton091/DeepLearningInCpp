#pragma once
#include"../../../Torch.hpp"
#include"../../../Tensor.hpp"
namespace torch{
namespace nn{
namespace Functional{
template<typename real>
Tensor<real> cross_entropy(const Tensor<real>& logits,const Tensor<real>& targets){
    if(logits.shape().size() != 2 || targets.shape().size() != 1 || targets.shape()[0] != logits.shape()[0]){ 
        throw std::invalid_argument("shape mismatch for cross_entropy.logits must be [batch_size,num_classes] and targets must be [batch_size]");
    }
    
    const int batch_size = logits.shape()[0];
    const int num_classes = logits.shape()[1];

    Tensor<real> logits_contig = logits.contiguous();
    Tensor<real> targets_contig = targets.contiguous();
    Tensor<real> softmax_probs(logits.shape(),false);
    real* logits_data = logits_contig.data_ptr();
    real* targets_data = targets_contig.data_ptr();
    real* softmax_data = softmax_probs.data_ptr();
    real total_loss = static_cast<real>(0.0);
    for(int i=0;i<batch_size;++i){
        int row_offset = i * num_classes;
        real max_logit = logits_data[row_offset];
        for(int j=row_offset+1;j<row_offset + num_classes;++j){
            if(logits_data[j] > max_logit) max_logit = logits_data[j];
        }
        real exp_sum = static_cast<real>(0.0);
        for(int j=row_offset;j<row_offset + num_classes;++j){
            exp_sum += std::exp(logits_data[j] - max_logit);
        }
        for(int j=row_offset;j<row_offset + num_classes;++j){
            softmax_data[j] = std::exp(logits_data[j] - max_logit) / exp_sum; 
        }

        int target_class = static_cast<int>(targets_data[i]);
        if(target_class < 0 || target_class >= num_classes){
            throw std::out_of_range("target class index out of range!");
        }

        real z_y = logits_data[row_offset + target_class];
        real loss_i = -z_y + max_logit + std::log(exp_sum);
        total_loss += loss_i;
    }
    total_loss /= static_cast<real>(batch_size);
    Tensor<real> output({},logits.requires_grad());
    output.data_ptr()[0] = total_loss;
    if(is_grad_enabled && logits.requires_grad()){
        output.set_requires_grad(true);
        auto grad_fn = std::make_shared<CrossEntropyBackward<real>>(logits_contig,softmax_probs,targets_contig);
        output.set_grad_fn(grad_fn);
    }
    return output;
}


}
}
}