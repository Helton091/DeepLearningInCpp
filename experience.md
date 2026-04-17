# 深度学习底层开发经验总结 (Experience & Lessons Learned)

## 1. 默认构造函数与 `delete` 陷阱 (Default Constructor & Deleted Functions)

### 报错现象：
```text
use of deleted function 'torch::AutogradMeta<float>::AutogradMeta()'
no matching function for call to 'torch::Tensor<float>::Tensor()'
```

### 病灶分析：
在 C++ 中，如果一个结构体（如 `AutogradMeta`）包含了一个自定义类成员（如 `Tensor<real> grad;`），当实例化这个结构体时，编译器会自动调用该成员的**默认无参构造函数**（`Tensor()`）。
但是，如果在 `Tensor` 类中我们显式定义了带参数的构造函数（如 `Tensor(shape, requires_grad)`），编译器就会**没收（隐式删除）**自动生成的无参构造函数。
这就导致了 `AutogradMeta` 在创建时，不知道该怎么初始化 `grad`，从而引发一连串的报错，最终连 `AutogradMeta` 的默认构造函数也被标记为 `deleted`。

### 解决方案：
必须在 `Tensor.hpp` 中显式地把默认构造函数要回来，并确保成员变量被正确初始化（避免野指针）：
```cpp
class Tensor {
private:
    std::shared_ptr<real []> data_ = nullptr; 
    std::vector<int> stride_; 
    std::vector<int> shape_; 
    size_t numel_ = 0; 
    std::shared_ptr<AutogradMeta<real>> autograd_meta_ = nullptr;
public:
    Tensor() = default; // 强制编译器生成默认构造函数
};
```

---

## 2. 构造函数参数匹配错误 (Constructor Signature Mismatch)

### 报错现象：
```text
no matching function for call to 'torch::Tensor<float>::Tensor(std::vector<int>&, std::vector<int>&, const std::shared_ptr<float []>&, const std::shared_ptr<float []>&, bool)'
```

### 病灶分析：
这是典型的“手滑”导致的低级错误。在编写视图操作（如 `squeeze`, `transpose`）返回新张量时，错误地把 `data_` 传入了两次：
```cpp
// 错误写法：多传了一个 data_
return Tensor<real>(new_shape, new_stride, data_, data_, autograd_meta_->requires_grad); 
```
而 `Tensor` 的构造函数签名只接受一个 `shared_data`：
```cpp
Tensor(const std::vector<int>& shape, std::vector<int>& stride, std::shared_ptr<real []> shared_data, bool requires_grad);
```
此外，在分配全新内存的 `contiguous()` 函数中，错误地传入了 `data_`（这会变成浅拷贝），而它本该调用只接收 `shape` 的构造函数来触发新内存的分配。

### 解决方案：
1. 删除多余的参数，严格对照头文件里的函数签名。
2. 访问 `requires_grad` 时，不要直接用 `autograd_meta_->requires_grad`，因为在纯前向张量中 `autograd_meta_` 可能是 `nullptr`，会引发段错误。应该使用封装好的安全 getter：`requires_grad()`。

```cpp
// 正确写法（视图操作，浅拷贝）
return Tensor<real>(new_shape, new_stride, data_, requires_grad());

// 正确写法（contiguous，深拷贝分配新内存）
Tensor<real> output(shape_, requires_grad());
```
