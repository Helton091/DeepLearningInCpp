#pragma once
#include"../../Torch.hpp"
#include"../../Tensor.hpp"
#include<vector>
#include<map>
#include<string>
namespace torch{
namespace nn{

template<typename real>
class Module{
protected:
    std::map<std::string, Tensor<real>> parameters_;
    std::map<std::string, std::shared_ptr<Module<real>>> modules_;
    bool is_training_ = true;
public:
    virtual ~Module() = default;
    bool is_training(){return is_training_;}
    Tensor<real>& register_parameter(const std::string& name, Tensor<real> param);

    template<typename ModuleType>
    std::shared_ptr<ModuleType> register_module(const std::string& name, std::shared_ptr<ModuleType> module);

    virtual std::map<std::string,Tensor<real>> named_parameters(const std::string& prefix = "",bool recursive = true) const;

    std::vector<Tensor<real>> parameters(bool recursive = true);

    virtual void zero_grad();

    virtual void train(bool on = true);

    virtual void eval();

    virtual Tensor<real> forward(const Tensor<real>& t){throw std::runtime_error("forward function isn't implemented!");}
};

template<typename real>
std::map<std::string,Tensor<real>> Module<real>::named_parameters(const std::string& prefix, bool recursive) const{
    std::map<std::string,Tensor<real>> result;
    for(const std::pair<const std::string, Tensor<real>>& kv : parameters_){
        const std::string& name = kv.first;
        const Tensor<real>& tensor = kv.second;
        std::string full_name = prefix.empty() ? name : prefix + "." + name;
        result[full_name] = tensor;
    }
    if(recursive){
        for(const std::pair<const std::string,std::shared_ptr<Module<real>>>& kv : modules_){
            std::string sub_prefix = prefix.empty() ? kv.first : prefix + "." + kv.first;
            std::map<std::string,Tensor<real>> sub_result = kv.second->named_parameters(sub_prefix,true);
            for(const std::pair<const std::string,Tensor<real>>& s : sub_result){
                result[s.first] = s.second;
            }
        }
    }
    return result;
}

template<typename real>
template<typename ModuleType>
std::shared_ptr<ModuleType> Module<real>::register_module(const std::string& name,std::shared_ptr<ModuleType> module){
    modules_[name] = module;
    return modules_[name];
}

template<typename real>
Tensor<real>& Module<real>::register_parameter(const std::string& name,Tensor<real> param){
    parameters_[name] = param;
    return parameters_[name];
}

template<typename real>
std::vector<Tensor<real>> Module<real>::parameters(bool recursive){
    std::vector<Tensor<real>> result_tensors;
    for(std::pair<const std::string, Tensor<real>>& kv : parameters_){
        result_tensors.push_back(kv.second);
    }
    if(recursive){
        for(std::pair<const std::string,std::shared_ptr<Module<real>>>& kv : modules_){
            std::vector<Tensor<real>> sub_tensors = kv.second->parameters();
            result_tensors.insert(result_tensors.end(),sub_tensors.begin(),sub_tensors.end());
        }
    }
    return result_tensors;
}

template<typename real>
void Module<real>::zero_grad(){
    for(std::pair<const std::string, Tensor<real>>& kv : parameters_){
        kv.second.zero_grad();
    }
    for(std::pair<const std::string,std::shared_ptr<Module<real>>>& kv : modules_){
        kv.second->zero_grad();
    }
}

template<typename real>
void Module<real>::eval(){
    train(false);
}

template<typename real>
void Module<real>::train(bool on){
    is_training_ = on;
    for(std::pair<const std::string,std::shared_ptr<Module<real>> >& kv : modules_){
        kv.second->train(on);
    }
}

}
}