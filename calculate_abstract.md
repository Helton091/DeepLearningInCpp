# Autograd Computational Graph Architecture

## 1. 算法视角：PyTorch 底层计算图的运行逻辑 (Algorithm Perspective)

为了让你彻底看清这套机制，我们用一个最经典的具体例子来做沙盘推演。
**假设我们要优化的数学模型是：$L = (A \times X + B) \times 2$**
- **$X$**：输入数据，不需要求导（`requires_grad=false`）。
- **$A$, $B$**：模型权重，需要求导（`requires_grad=true`），它们是**叶子节点（Leaf Tensors）**。
- 我们分步执行前向代码：
  1. `M = A * X` (矩阵乘法)
  2. `Y = M + B` (加法，带有广播)
  3. `L = Y * 2` (标量乘法)

### 阶段一：前向传播与“埋点”（Forward Pass & Graph Construction）
当你的 C++ 代码依次执行这三行时，底层不仅在算数字，还在搭建下面这张 DAG（有向无环图）：

1. **算 `M = A * X` 时**：
   - 因为 $A$ 需要求导，所以输出 $M$ 也被传染了 `requires_grad=true`。
   - 系统 `new` 了一个节点 `MatmulBackward`。
   - `MatmulBackward` 偷偷保存了 $A$ 和 $X$ 的指针（或者副本），因为将来算偏导时要用到。
   - $M$ 的 `grad_fn` 指针，指向了这个 `MatmulBackward`。

2. **算 `Y = M + B` 时**：
   - 系统 `new` 了一个节点 `AddBackward`。
   - `AddBackward` 偷偷保存了 $M$ 和 $B$ 的指针。
   - $Y$ 的 `grad_fn` 指针，指向了这个 `AddBackward`。

3. **算 `L = Y * 2` 时**：
   - 系统 `new` 了一个节点 `MulScalarBackward`。
   - `MulScalarBackward` 偷偷保存了 $Y$ 的指针，并记下了常数 `2`。
   - $L$ 的 `grad_fn` 指针，指向了这个 `MulScalarBackward`。

**此时，计算图已经建好了！它的形状像一根链条：**
`L` $\rightarrow$ `MulScalarBackward` $\rightarrow$ `Y` $\rightarrow$ `AddBackward` $\rightarrow$ `M` $\rightarrow$ `MatmulBackward` $\rightarrow$ `A`

### 阶段二：拓扑排序（Topological Sort）
当你执行 `L.backward()` 时，Autograd 引擎启动。
在这个简单的线性例子里，拓扑排序的顺序就是沿着 `grad_fn` 一路往回找：
1. `MulScalarBackward`
2. `AddBackward`
3. `MatmulBackward`

*(注：如果你的网络有像 ResNet 里的 Skip Connection 分支，比如 $Y = M + M$，那么 $M$ 会被两个节点依赖，拓扑排序就会确保 $M$ 必须等这两个节点都把梯度传过来之后，才把自己激活向下传。)*

### 阶段三：反向传播与链式法则（Backward Pass & Chain Rule）
引擎开始按照拓扑排序的顺序，依次激活节点，并传递梯度（`grad_output`）。
一开始，系统强行给终点 $L$ 生成一个全是 `1.0` 的初始梯度 $\frac{\partial L}{\partial L}$，作为第一棒传下去。

1. **激活 `MulScalarBackward` (它手里的 `grad_output` 是 `1.0`)**
   - 它的数学逻辑是 $L = Y \times 2$，所以对 $Y$ 的局部偏导 $\frac{\partial L}{\partial Y} = 2$。
   - 链式法则：它算出的传递梯度 = `grad_output` $\times$ 局部偏导 = `1.0 * 2 = 2.0`。
   - 它把这块全是 `2.0` 的梯度，丢给它手里攥着的 $Y$。

2. **激活 `AddBackward` (它手里的 `grad_output` 是刚收到的 `2.0`)**
   - 它的数学逻辑是 $Y = M + B$，对 $M$ 和 $B$ 的局部偏导都是 `1`。
   - 链式法则传给 $M$：`grad_output * 1 = 2.0`。把它丢给 $M$。
   - 链式法则传给 $B$：`grad_output * 1 = 2.0`。
   - **关键拦截**：因为 $B$ 是叶子节点，系统直接把这块 `2.0` 的梯度累加到 `B.grad` 里！至此，$B$ 的梯度计算大功告成！

3. **激活 `MatmulBackward` (它手里的 `grad_output` 也是 `2.0`)**
   - 它的数学逻辑是 $M = A \times X$。
   - 链式法则传给 $A$：根据矩阵求导法则，$\frac{\partial L}{\partial A} = \text{grad\_output} \times X^T$。
   - 它用手里攥着的 $X$ 的转置去乘以 `2.0` 的梯度，算出一块新矩阵。
   - 因为 $A$ 是叶子节点，它把这块新矩阵累加到 `A.grad` 里。
   - 传给 $X$ 呢？因为一开始 $X$ 设置了 `requires_grad=false`，引擎一看，直接丢弃，不浪费算力。

**至此，引擎停机，`A.grad` 和 `B.grad` 完美生成！这就是 Autograd 的全部魔法。**

---

## 3. 当前项目实装架构与开发路线图 (Current Implementation Roadmap)

基于你当前实际的 C++ 代码结构，我们重新梳理了目前的架构分布。这与之前抽象的教程稍有不同，你的文件拆分更加细化（比如抽离了 `TorchBackwardFunctions.hpp`），这是非常好的底层工程实践。

### 3.1 当前文件结构与组件分布

1. **`Tensor.hpp` (核心声明)**
   - 包含 `AutogradMeta` 和 `BackwardFunction` 的前向声明。
   - `Tensor` 类新增了 `autograd_meta_` 指针。
   - 声明了相关的 Autograd 接口：`grad()`, `add_grad()`, `grad_fn()`, `set_grad_fn()`, `is_leaf()`, `backward()`。

2. **`Torch/AutoGrad/AutoGrad.hpp` (自动微分基础数据结构)**
   - 定义了 `AutogradMeta` 结构体（包含 `grad`, `has_grad`, `grad_fn`, `is_lead`, `requires_grad`）。
   - **[已完成 ✅]** 实现了核心反向广播降维算子 `unbroadcast`。

3. **`Tensor/TensorUnaryOps.hpp` (一元操作)**
   - **[已完成 ✅]** 实现了 `sum(int dim, bool keep_dim)`，这是 `unbroadcast` 的基石。

4. **`Torch/AutoGrad/TorchBackwardFunctions.hpp` (反向节点与接口实现)**
   - 定义了 `BackwardFunction` 基类。
   - 定义了具体的加法操作节点 `AddBackward`（目前只有构造函数）。
   - 实现了 `Tensor` 类的 Autograd 接口（如 `is_leaf()`, `grad_fn()`, `grad()`）。

---

### 3.2 接下来需要手敲的路线图 (Next Steps for Tensor Addition)

为了让加法（`A + B`）的计算图真正跑起来，你需要按顺序完成以下任务，建议你对照着去修改代码：

#### Step 1: 完善 `Tensor` 的 Autograd 接口实现 (`TorchBackwardFunctions.hpp`)
你目前只实现了几个接口，且有些许瑕疵，需要修复和补全：
- **`is_leaf()` 的修复**：目前返回的是 `requires_grad`，这是不对的。应该返回 `autograd_meta_->is_lead` (另外，建议你在 `AutoGrad.hpp` 里把 `is_lead` 拼写更正为 `is_leaf`)。
- **`set_grad_fn()` 的签名与实现**：在 `Tensor.hpp` 中，你声明的是 `void set_grad_fn(const BackwardFunction<real>& fn);`。它应该改为接收智能指针：`void set_grad_fn(std::shared_ptr<BackwardFunction<real>> fn);`。并在实现中，将 `autograd_meta_->is_leaf` 设为 `false`。
- **`add_grad()` 的实现**：接收梯度 `g`。如果 `has_grad == false`，那么 `grad = g` 且 `has_grad = true`；如果已经有梯度了，那么 `grad = grad + g`。
- **`backward()` 的空壳**：暂时留空，或者只打印一行 "backward called"，留到最后再写。

#### Step 2: 实现 `AddBackward::apply` (`TorchBackwardFunctions.hpp`)
在 `AddBackward` 类中，重写 `apply` 方法：
- 调用你已完成的 `unbroadcast(grad_output, tensor_a_.shape())`。
- 将 unbroadcast 后的结果，传给 `tensor_a_.add_grad(...)` 和 `tensor_b_.add_grad(...)`。

#### Step 3: 前向操作的“埋点建图” (`TensorBinaryOps.hpp`)
去找到你写好的 `operator+` 函数。
- 算完数值 `output` 之后，检查 `A.requires_grad()` 或 `B.requires_grad()`。
- 如果为 true，需要为 `output` 初始化一个 `AutogradMeta`（你可以写个辅助函数或者直接分配）。
- `new` 一个 `AddBackward` 节点。
- 用 `output.set_grad_fn(grad_fn)` 把节点塞进 `output`。

#### Step 4: 编写拓扑排序与 `backward()` 引擎 (`TorchBackwardFunctions.hpp`)
- 在真正实现 `Tensor::backward()` 时，你需要用一个队列/栈进行图的遍历。
- 确保子节点的梯度全部收集完毕后，再调用 `grad_fn->apply(grad())` 向下传递。

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