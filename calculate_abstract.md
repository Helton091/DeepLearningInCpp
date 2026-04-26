# Autograd Computational Graph Architecture

## 0. 数学基础：张量对张量的导数与 VJP 魔法 (Tensor Calculus & VJP)

在进入代码和算法之前，我们必须回答一个直击灵魂的数学问题：“张量对张量的偏导数到底是个什么鬼？”
如果你习惯了标量微积分 $y = f(x)$ 的导数 $dy/dx$，当你面对一个矩阵 $Y$ 对另一个矩阵 $X$ 求导时，事情会变得极其恐怖。

### 0.1 严格的数学定义：雅可比矩阵 (Jacobian Matrix)
假设有一个函数 $\vec{y} = f(\vec{x})$，其中 $\vec{x}$ 是长度为 $n$ 的向量，$\vec{y}$ 是长度为 $m$ 的向量。
在数学上，“向量 $\vec{y}$ 对向量 $\vec{x}$ 的导数”被称为**雅可比矩阵（Jacobian）** $J$。它是一个 $m \times n$ 的矩阵，矩阵中的每一个元素是 $\vec{y}$ 的每一个分量对 $\vec{x}$ 的每一个分量的偏导数：
$
J = \begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{pmatrix}
$
如果 $Y$ 是一张 $200 \times 200$ 的图像（$40000$ 个元素），$X$ 也是 $200 \times 200$。那么 $\frac{\partial Y}{\partial X}$ 的雅可比矩阵将是一个 $40000 \times 40000$ 的超大矩阵（包含 16 亿个偏导数）！
如果在神经网络的反向传播中，我们在节点之间传递这种雅可比矩阵，任何显卡都会瞬间内存溢出（OOM）。

### 0.2 深度学习的救命稻草：标量损失假定
为什么 PyTorch 不会 OOM？为什么我们传递的 `grad_output` 形状总是和张量自己一样大？
因为**深度学习的最终目标（Loss）永远是一个标量（Scalar）！**

在 Autograd 中，我们**永远不计算**“中间张量 $Y$ 对中间张量 $X$ 的导数”。
我们真正在计算和传递的，永远是**“最终的标量损失 $L$ 对中间张量 $X$ 的导数”**，记为 $\nabla_X L$ 或 $\frac{\partial L}{\partial X}$。
因为 $L$ 是标量，所以根据微积分，$\frac{\partial L}{\partial X}$ 的形状永远与 $X$ 的形状一模一样！

### 0.3 引擎的灵魂：向量-雅可比乘积 (Vector-Jacobian Product, VJP)
那么，链式法则怎么体现呢？
假设有一条链：$X \xrightarrow{f} Y \xrightarrow{g} L$ ($L$ 是标量)。
根据链式法则：
$ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial X} $
在我们的引擎中：
- $\frac{\partial L}{\partial Y}$ 就是从父节点传下来的 `grad_output`（它和 $Y$ 的形状一样，我们可以把它拉平看作一个行向量 $\vec{v}^T$）。
- $\frac{\partial Y}{\partial X}$ 就是那个恐怖的雅可比矩阵 $J$。
- 所以，节点要计算传给下一个节点的梯度，实际上是在算 **$\vec{v}^T \cdot J$**！

**魔法来了：计算 $\vec{v}^T \cdot J$ 根本不需要把 $J$ 完整地写出来！**
例如对于加法 $Y = X + B$：
雅可比矩阵 $J$ 本质上是个单位矩阵 $I$。所以 $\vec{v}^T \cdot I = \vec{v}^T$。
这就是为什么在 `AddBackward` 中，传给 $X$ 的梯度等于 `grad_output`，完全不需要构造什么矩阵，直接把传进来的梯度丢下去就行了。

对于矩阵乘法 $M = A \times X$：
数学推导可以证明，标量 $L$ 对 $A$ 的偏导数 $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial M} \times X^T$。
这就是你在 `MatmulBackward` 里要写的代码：`grad_A = matmul(grad_output, X.transpose())`。

**结论**：你在编写 `BackwardFunction::apply` 时，你写的代码本质上就是在实现这个算子的 **VJP（Vector-Jacobian Product）解析解**。你利用算子的数学特性，直接用 `grad_output` 和输入张量算出了结果，巧妙地避开了构造雅可比矩阵的灾难。

---

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

#### Step 4: 编写拓扑排序与 `backward()` 引擎 (The Topological Sort Engine)

**为什么不能用简单的递归（DFS）？**
如果你在 `AddBackward::apply` 里直接调用 `tensor_a_.backward()`，这是一种深度优先搜索（DFS）。对于直线型网络没有问题。但如果你实现了像 **ResNet** 这样的网络，存在残差连接（Skip Connection）：$Y = M + A$ 和 $Z = M + B$，最后 $L = Y + Z$。
此时 $M$ 被使用了两次。如果用 DFS，$L$ 调 $Y$，$Y$ 调 $M$，$M$ 就会立刻把不完整的梯度（只包含了 $Y$ 的部分）传给更底层！这是极其严重的错误。
**正确的做法：** $M$ 必须等 $Y$ 和 $Z$ **都**把梯度传给它之后，才能激活自己向下传。这就是**拓扑排序（Topological Sort）**的入度（In-degree）机制。

**实现拓扑排序的前提（核心接口）：**
为了让引擎能在图上游走，你的 `BackwardFunction` 必须能告诉引擎：“我的输入是谁？”。
因此，你需要去基类中新增一个纯虚函数接口：
```cpp
// 它的作用是向引擎报告自己的依赖项
virtual std::vector<Tensor<real>> get_inputs() const = 0; 
```
你需要在 `AddBackward` 中实现它，简单地返回 `{tensor_a_, tensor_b_}` 即可。

**引擎设计的两大阶段（算法思想）：**

1. **阶段 1：图遍历，计算所有节点的入度 (In-degree)**
   - **目标**：找出图里每一个节点被多少个父节点依赖。
   - **数据结构 `std::unordered_map<AutogradMeta<real>*, int> in_degree`**：
     这是 C++ 里的哈希表（Hash Map）。它的键（Key）是你张量里那个唯一的 `AutogradMeta` 裸指针，值（Value）是它的入度计数。我们用裸指针作为唯一 ID，因为不同的 Tensor 副本共享同一个 Meta。
   - **数据结构 `std::unordered_set<AutogradMeta<real>*> visited`**：
     哈希集合，用来记录哪些节点已经被遍历过了，防止在环状或菱形结构中重复遍历。
   - **数据结构 `std::queue<Tensor<real>> bfs_queue`**：
     广度优先搜索队列。
   - **流程**：把起点（调用 `backward` 的那个 Tensor）放进队列。不断弹出节点，通过它的 `grad_fn` 和 `get_inputs()` 找到所有的子节点（输入张量）。给这些子张量的入度 +1，如果它们没被访问过，就放进队列继续遍历。

2. **阶段 2：执行反向传播 (收割梯度)**
   - **目标**：按照入度归零的顺序，依次激活节点，向下传递梯度。
   - **数据结构 `std::queue<Tensor<real>> ready_queue`**：
     这是执行队列。**只有入度为 0 的节点，才有资格进入这个队列！**
   - **流程**：
     1. 把起点放入 `ready_queue`（起点的入度天然是 0）。
     2. 只要队列不空，弹出一个节点 `curr`。
     3. 调用它的 `curr.grad_fn()->apply(curr.grad())`，把梯度向下传！
     4. 遍历它的所有子节点（输入张量），在哈希表 `in_degree` 里把它们的入度减 1。
     5. **灵魂判定**：如果某个子节点的入度减到了 0，说明它所有的父节点都已经把梯度传给它了！把它推入 `ready_queue`，等待下一次激活。

这就是工业级 Autograd 引擎的全部奥秘！不需要我写代码，有了这套数据结构和算法流程，你完全可以自己手搓出来！

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

---

## 4. C++ 语法弹药库 (Syntax Cheat Sheet)

在手敲拓扑排序时，你可能会高频用到以下 C++ 特性：

### 4.1 `std::unordered_map` 的常用操作
它是基于哈希表实现的字典，查找和插入时间复杂度平均为 $O(1)$。
**重点解答：** 因为我们的 Key 是裸指针 (`AutogradMeta<real>*`)，C++ 标准库 `<functional>` 原生提供了 `std::hash<T*>` 的特化版本，会直接利用指针的内存地址计算哈希值。因此，**你完全不需要手动提供哈希函数**！

```cpp
std::unordered_map<AutogradMeta<real>*, int> in_degree;

// 1. 计数加一（C++很聪明，如果键不存在，会自动初始化为0再加1）
in_degree[meta_ptr]++; 

// 2. 计数减一
in_degree[meta_ptr]--;

// 3. 检查某个键是否存在
if (in_degree.find(meta_ptr) != in_degree.end()) {
    // 存在
}
```

### 4.2 `std::unordered_set` 的常用操作
用来记录哪些节点已经访问过，防止死循环。

```cpp
std::unordered_set<AutogradMeta<real>*> visited;

// 1. 插入并判断是否是第一次插入
if (visited.insert(meta_ptr).second) {
    // 插入成功，说明之前没访问过
} else {
    // 之前已经访问过了
}
```

### 4.3 `std::queue` 的常用操作
用于 BFS 遍历和执行队列。

```cpp
std::queue<Tensor<real>> q;

q.push(tensor);            // 入队（放到队尾）
Tensor<real> curr = q.front(); // 查看队首元素
q.pop();                   // 弹出队首元素（注意：pop本身不返回值）
bool is_empty = q.empty(); // 判断队列是否为空
```

### 4.4 智能指针获取裸指针
你的哈希表需要用 `AutogradMeta<real>*` 作为唯一键。

```cpp
// 假设你有一个 std::shared_ptr<AutogradMeta<real>>

---

## 5. 进阶算子：Conv2d (二维卷积) 的数学与实现逻辑

在实现了基础算子之后，`Conv2d` 是深度学习框架中最核心的计算机视觉算子。由于卷积操作本质上可以转化为矩阵乘法（通过 `im2col` 技术），我们在这里详细解析它的前向与反向传播数学逻辑。

### 5.1 前向传播 (Forward Pass)

假设我们有一个输入张量 $X$ 和卷积核权重 $W$（为简化推导，暂不考虑 bias，bias 的逻辑与全连接层一致，直接加在输出通道上即可）。

- **输入 $X$ 形状**: `[N, C_in, H_in, W_in]`
  - $N$: Batch Size
  - $C_{in}$: 输入通道数
  - $H_{in}, W_{in}$: 输入图像的高和宽
- **权重 $W$ 形状**: `[C_out, C_in, K_H, K_W]`
  - $C_{out}$: 输出通道数
  - $K_H, K_W$: 卷积核的高和宽
- **输出 $Y$ 形状**: `[N, C_out, H_out, W_out]`

**朴素实现 (Naive Sliding Window)**：
通过七层嵌套循环（N, C_out, H_out, W_out, C_in, K_H, K_W）直接计算。虽然代码简单，但在 CPU/GPU 上执行效率极低，因为无法利用矩阵乘法（BLAS）的高度优化。

**工业级实现 (im2col + GEMM)**：
1. **im2col (Image to Column)**：将输入张量 $X$ 中每次滑动窗口覆盖的局部区域（形状为 `[C_in, K_H, K_W]`）展平为一个一维向量（长度为 `C_in * K_H * K_W`）。滑动窗口在图像上移动 `H_out * W_out` 次，我们将这些向量按列拼接，形成一个大矩阵 $X_{col}$。
   - $X_{col}$ 的形状为 `[C_in * K_H * K_W, H_out * W_out]`。（对于多 Batch，通常在 Batch 维度循环，或者展平到列中）。
2. **Reshape 权重**：将权重 $W$ 展平为二维矩阵 $W_{row}$。
   - $W_{row}$ 的形状为 `[C_out, C_in * K_H * K_W]`。
3. **矩阵乘法 (GEMM)**：
   - $Y_{col} = W_{row} \times X_{col}$
   - $Y_{col}$ 的形状为 `[C_out, H_out * W_out]`。
4. **col2im / Reshape**：将 $Y_{col}$ 变回 `[C_out, H_out, W_out]`，再加上 Batch 维度，得到最终输出 $Y$。

### 5.2 反向传播 (Backward Pass)

在反向传播时，`Conv2dBackward` 节点会接收到来自上游的梯度 `grad_output`（即 $\frac{\partial L}{\partial Y}$），其形状与输出 $Y$ 相同：`[N, C_out, H_out, W_out]`。我们需要计算对权重 $W$ 的梯度（$\frac{\partial L}{\partial W}$）和对输入 $X$ 的梯度（$\frac{\partial L}{\partial X}$）。

基于 `im2col` 的前向传播可以看作是一个矩阵乘法 $Y_{col} = W_{row} \times X_{col}$。根据我们在 0.3 节学到的矩阵乘法 VJP 规则，卷积的反向传播可以极其优雅地转化为：

#### 1. 对权重 $W$ 的求导 ($\nabla_W L$)
根据矩阵乘法求导法则：$\nabla_{W_{row}} L = \nabla_{Y_{col}} L \times X_{col}^T$
- $\nabla_{Y_{col}} L$ 是 `grad_output` 展平后的矩阵，形状为 `[C_out, H_out * W_out]`。
- $X_{col}$ 是前向传播时保存的 im2col 结果，形状为 `[C_in * K_H * K_W, H_out * W_out]`。转置后为 `[H_out * W_out, C_in * K_H * K_W]`。
- 矩阵相乘得到形状为 `[C_out, C_in * K_H * K_W]` 的梯度。
- 最后将其 Reshape 回 `[C_out, C_in, K_H, K_W]`，加上如果有多个 Batch 则进行累加，即可得到 `W.grad`。

#### 2. 对输入 $X$ 的求导 ($\nabla_X L$)
根据矩阵乘法求导法则：$\nabla_{X_{col}} L = W_{row}^T \times \nabla_{Y_{col}} L$
- $W_{row}^T$ 的形状为 `[C_in * K_H * K_W, C_out]`。
- 相乘后得到形状为 `[C_in * K_H * K_W, H_out * W_out]` 的 $\nabla_{X_{col}} L$。
- **难点：col2im**。因为在 `im2col` 过程中，由于滑动窗口的重叠，输入图像的同一个像素可能被复制到了 $X_{col}$ 的多个不同列中。因此，在反向传播时，我们需要把 $\nabla_{X_{col}} L$ 中的梯度累加（scatter_add）回原始图像尺寸 `[C_in, H_in, W_in]` 对应的位置。这个操作被称为 `col2im`。

### 5.3 在框架中的落地思路 (Roadmap for Conv2d)

1. **基础张量操作**：
   - 首先，你需要实现一个可靠的 `im2col` 函数和对应的逆操作 `col2im` 函数。这两个函数通常是纯 C++ 的指针/索引操作，需要仔细处理 `stride` 和 `padding`。
2. **前向算子 `conv2d`**：
   - 接受 `X`, `W`, `bias`, `stride`, `padding`。
   - 调用 `im2col` -> 调整形状 -> 执行你现有的 `matmul` -> 加上 `bias` -> 还原形状。
   - 埋点：创建 `Conv2dBackward` 节点，保存 $X$（或者直接保存算好的 $X_{col}$ 牺牲内存换速度）、$W$ 的指针以及 stride/padding 参数。
3. **反向节点 `Conv2dBackward`**：
   - 接收 `grad_output`。
   - 计算 $W$ 梯度：对 `grad_output` reshape，与缓存的 $X_{col}^T$ 做 `matmul`。
   - 计算 $X$ 梯度：权重转置与 `grad_output` 做 `matmul`，然后调用 `col2im` 将结果累加回原图尺寸。
   - 分发梯度到 `add_grad`。
auto meta_ptr = tensor.autograd_meta_.get(); // .get() 可以提取底层的裸指针
```
