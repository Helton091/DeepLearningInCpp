#pragma once
#include"Module.hpp"
#include"../../Tensor.hpp"
#include"../../Torch.hpp"
#include<vector>
namespace torch{
namespace nn{
template<typename real>
class Sequential : public Module<real>{
private:
    std::vector<std::shared_ptr<Module<real>>> ordered_modules_;
    std::string get_next_name(){return std::to_string(ordered_modules_.size());}
public:
    Sequential() = default;
    
    template<typename... Modules>
    Sequential(std::shared_ptr<Modules>... mods){
        (push_back(mods),...);
    }

    void push_back(std::shared_ptr<Module<real>> module){
        std::string name = get_next_name();
        // register_module 是模板成员函数，由于 module 的类型是 std::shared_ptr<Module<real>>
        // 编译器可以推导出 ModuleType = Module<real>，但为了保险起见，可以显式写出：
        this->template register_module<Module<real>>(name, module);
        ordered_modules_.push_back(module);
    }

    Tensor<real> forward(const Tensor<real>& x) override{
        Tensor<real> output = x;
        for(std::shared_ptr<Module<real>> m : ordered_modules_){
            output = m->forward(output);
        }
        return output;
    }


};


}
}