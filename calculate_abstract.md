# Autograd Computational Graph Architecture

## 1. 算法视角：PyTorch 底层计算图的运行逻辑 (Algorithm Perspective)

在深度学习中，当我们写下 `loss.backward()` 时，底层到底发生了什么？让我们抛开代码，先从纯算法的角度，像剥洋葱一样看清计算图（Computational Graph）的生命周期。它分为三个核心阶段：

### 阶段一：前向传播与“埋点”（Forward Pass & Graph Construction）
当你执行 $C = A + B$ 时，系统不仅算出了 $C$ 的数值，还在后台偷偷画了一张有向无环图（DAG）：
1. **创建节点**：系统会创建一个代表“加法求导”的节点（比如叫 `AddBackward`）。
2. **记录输入**：这个节点会把 $A$ 和 $B$ 的引用（或者计算梯度必需的中间变量）死死攥在手里。
3. **连接边**：系统把 $C$ 标记为由 `AddBackward` 创造出来的。
4. **叶子节点**：像 $A$ 和 $B$ 这种由用户直接创建、需要求导的张量，被称为“叶子节点（Leaf Tensors）”。它们没有创造者。

### 阶段二：拓扑排序（Topological Sort）
当用户对最终的标量 $L$ 调用 `L.backward()` 时，系统首先面临一个数学危机：多条路径的梯度如何正确累加？
例如：$D = A \times 2$, $E = A + 3$, $L = D + E$。
如果系统先算 $D$ 给 $A$ 的梯度，直接把梯度加给 $A$，这没问题。但此时 $A$ 还没收到 $E$ 传来的梯度！如果此时 $A$ 觉得自己梯度已经算完了，去触发了别的东西，那就全错了。
**解法**：系统必须从 $L$ 开始，逆向遍历整张图，计算每个节点的“入度”（即它有多少个子节点）。只有当一个节点收齐了**所有**子节点传来的梯度后，它才能把自己激活，去算传给上一层的梯度。这就是拓扑排序。

### 阶段三：反向传播与链式法则（Backward Pass & Chain Rule）
引擎按照拓扑排序的顺序，依次激活每个 `Backward` 节点。
当 `AddBackward` (创造了 $C$) 被激活时，它会收到上一层传来的 $\frac{\partial L}{\partial C}$（我们称之为 `grad_output`）。
它的任务只有两个：
1. **算局部偏导**：加法对 $A$ 和 $B$ 的局部偏导都是 1。
2. **应用链式法则**：用 `grad_output` $\times$ 局部偏导，算出传给 $A$ 的梯度（$\frac{\partial L}{\partial A}$）和传给 $B$ 的梯度（$\frac{\partial L}{\partial B}$）。
3. **分发**：把算出来的梯度，累加到 $A$ 和 $B$ 的 `.grad` 属性中。

这就是整个 Autograd 的宏观算法！

---

## 2. C++ 视角：对应的底层实现细节 (C++ Implementation)

理解了算法，我们来看看如何用 C++ 的面向对象特性，把这三个阶段落地到你的项目中。

### 2.1 核心组件对应关系
| 算法概念 | C++ 实现 |
|---|---|
| 计算图节点 | `class BackwardFunction` 的子类（如 `AddBackward`） |
| 张量的梯度存储 | `Tensor` 内的 `AutogradMeta::grad` |
| 连接节点的边 | `Tensor` 内的 `AutogradMeta::grad_fn` 指针 |
| 链式法则计算 | `BackwardFunction::apply(const Tensor& grad_output)` 方法 |

### 2.2 数据结构定义 (`Tensor.hpp` & `AutoGrad.hpp`)

为了让所有的 `Tensor` 副本（由于按值传递）能共享同一个梯度和计算图节点，我们使用了 **Pimpl（共享状态）模式**。

```cpp
// in AutoGrad.hpp
namespace torch {

// 所有反向计算节点的基类
template<typename real>
class BackwardFunction {
public:
    virtual ~BackwardFunction() = default;
    // 核心引擎接口：接收上一层传来的梯度，算完后分发给自己的输入
    virtual void apply(const Tensor<real>& grad_output) = 0;
};

// 张量的“内鬼”，存储计算图状态
template<typename real>
struct AutogradMeta {
    Tensor<real> grad;             // 存储梯度
    bool has_grad = false;         // 标记梯度是否已初始化
    std::shared_ptr<BackwardFunction<real>> grad_fn = nullptr; // 边：指向创造自己的节点
    bool is_leaf = true;           // 叶子节点标记
    bool requires_grad = false;    // 是否需要求导
};

}

// in Tensor.hpp
template<typename real>
class Tensor {
private:
    // ... data_, shape_, stride_ ...
    // 共享的 Autograd 状态
    std::shared_ptr<AutogradMeta<real>> autograd_meta_ = nullptr;
    // ...
};
```

### 2.3 阶段一：前向传播埋点 (`TensorBinaryOps.hpp`)

以 $C = A + B$ 为例，我们需要在算出 $C$ 的数值后，立刻把图建起来。

```cpp
// 具体的加法反向节点（继承自 BackwardFunction）
template<typename real>
class AddBackward : public BackwardFunction<real> {
private:
    Tensor<real> tensor_a_; // 死死攥住 A 和 B，为了将来能找到它们分发梯度
    Tensor<real> tensor_b_;

public:
    AddBackward(const Tensor<real>& a, const Tensor<real>& b) : tensor_a_(a), tensor_b_(b) {}

    // 阶段三：链式法则的具体实现
    void apply(const Tensor<real>& grad_output) override {
        // C = A + B，所以 dC/dA = 1。dL/dA = grad_output * 1
        if (tensor_a_.requires_grad()) {
            // 【难点】：如果 A 参与了广播，必须把 grad_output 沿着广播维度 sum 掉！
            // Tensor<real> grad_a = unbroadcast(grad_output, tensor_a_.shape());
            // tensor_a_.add_grad(grad_a); 
        }
        // 对 B 同理...
    }
};

// 前向埋点
template<typename real>
Tensor<real> operator+(const Tensor<real>& A, const Tensor<real>& B) {
    // 1. 正常的数值计算... 得到 output
    
    // 2. 前向埋点：建图！
    if (A.requires_grad() || B.requires_grad()) {
        // 创建节点
        auto grad_fn = std::make_shared<AddBackward<real>>(A, B);
        // 连接边
        output.set_grad_fn(grad_fn);
    }
    
    return output;
}
```

### 2.4 阶段二与三的触发：`backward()` 引擎
当用户调用 `tensor.backward()` 时，C++ 引擎必须执行拓扑排序并调用 `apply`。
（为了降低难度，在早期版本中，如果你的网络是线性的（没有复用的分支），你可以用简单的递归调用 `grad_fn->apply()` 来代替复杂的拓扑排序引擎。但最终一定要上拓扑排序队列。）