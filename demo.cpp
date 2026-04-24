#include "Torch.hpp"
#include <iostream>

using std::cout;
using std::endl;
using namespace torch;

void print_separator(const std::string& title) {
    cout << "\n========================================\n"
         << "   " << title << "\n"
         << "========================================\n";
}

int main() {
    print_separator("Test 1: 加减乘除大杂烩 (Complex Arithmetic)");
    // 数学公式: L = ((A + B) * C - D) / 2.0
    // A=1, B=2, C=3, D=4
    auto A = torch::ones<float>({2, 2}, true);
    auto B = torch::ones<float>({2, 2}, true); B.fill_(2.0f);
    auto C = torch::ones<float>({2, 2}, true); C.fill_(3.0f);
    auto D = torch::ones<float>({2, 2}, true); D.fill_(4.0f);

    auto t1 = A + B;       // t1 = 3
    auto t2 = t1 * C;      // t2 = 9
    auto t3 = t2 - D;      // t3 = 5
    auto L = t3 / 2.0f;    // L = 2.5

    cout << "L (Expected: 2.5):\n" << L << "\n";
    
    // Retain gradients for intermediate tensors for debugging
    t1.set_requires_grad(true);
    t2.set_requires_grad(true);
    t3.set_requires_grad(true);

    L.backward();

    cout << "DEBUG - t3.grad (Expected: 0.5):\n" << t3.grad() << "\n";
    cout << "DEBUG - t2.grad (Expected: 0.5):\n" << t2.grad() << "\n";
    cout << "DEBUG - t1.grad (Expected: 1.5):\n" << t1.grad() << "\n";

    // 手算梯度 (dL/d...)
    // dL/dt3 = 1/2 = 0.5
    // dL/dD = dL/dt3 * dt3/dD = 0.5 * (-1) = -0.5
    // dL/dt2 = dL/dt3 * dt3/dt2 = 0.5 * 1 = 0.5
    // dL/dC = dL/dt2 * dt2/dC = 0.5 * t1 = 0.5 * 3 = 1.5
    // dL/dt1 = dL/dt2 * dt2/dt1 = 0.5 * C = 0.5 * 3 = 1.5
    // dL/dA = dL/dt1 * dt1/dA = 1.5 * 1 = 1.5
    // dL/dB = dL/dt1 * dt1/dB = 1.5 * 1 = 1.5

    cout << "A.grad (Expected: 1.5):\n" << A.grad() << "\n";
    cout << "B.grad (Expected: 1.5):\n" << B.grad() << "\n";
    cout << "C.grad (Expected: 1.5):\n" << C.grad() << "\n";
    cout << "D.grad (Expected: -0.5):\n" << D.grad() << "\n";

    print_separator("Test 2: 非交换标量运算 (Non-commutative Scalar Ops)");
    // 数学公式: L = 10.0 / (5.0 - X)
    // X=3. L = 10 / 2 = 5
    auto X = torch::ones<float>({2, 2}, true); X.fill_(3.0f);
    
    auto denom = 5.0f - X; // denom = 2
    auto L2 = 10.0f / denom;
    
    cout << "L2 (Expected: 5.0):\n" << L2 << "\n";
    L2.backward();

    // 手算梯度 (dL2/dX)
    // dL2/ddenom = -10 / denom^2 = -10 / 4 = -2.5
    // ddenom/dX = -1
    // dL2/dX = (-2.5) * (-1) = 2.5
    cout << "X.grad (Expected: 2.5):\n" << X.grad() << "\n";

    print_separator("Test 3: 带有广播机制的除法与减法 (Broadcast Div/Sub)");
    // X: [2, 3], 值全为 12
    // Y: [3],    值全为 3
    auto X_b = torch::ones<float>({2, 3}, true); X_b.fill_(12.0f);
    auto Y_b = torch::ones<float>({3}, true);    Y_b.fill_(3.0f);

    // Z = X_b / Y_b - Y_b
    // Z = 12 / 3 - 3 = 4 - 3 = 1
    auto div_res = X_b / Y_b;
    auto Z = div_res - Y_b;
    
    cout << "Z (Expected: 1.0):\n" << Z << "\n";
    Z.backward();

    // 手算梯度
    // dZ/ddiv_res = 1. 所以 dZ/dX_b = 1 * (1/Y_b) = 1/3
    // X_b 形状是 [2, 3]，不需要 unbroadcast，直接全是 1/3 (0.333)
    cout << "X_b.grad (Expected: 0.3333):\n" << X_b.grad() << "\n";

    // 对 Y_b 的梯度有两部分来源：
    // 1. 减号右边: dZ/dY_b (right) = -1
    // 2. 除号右边: dZ/dY_b (left) = dZ/ddiv_res * ddiv_res/dY_b = 1 * (-X_b / Y_b^2) = -12/9 = -4/3
    // 聚合后：总偏导 = -1 - 4/3 = -7/3 = -2.3333
    // 因为 Y_b 参与了 broadcast (被扩展成了两行)，所以它的真实梯度必须 sum 起来：
    // Y_b.grad = (-7/3) * 2 = -14/3 = -4.6666
    cout << "Y_b.grad (Expected: -4.6666):\n" << Y_b.grad() << "\n";

    print_separator("Test 4: 全连接层 (Linear Layer w/ Matmul Backward)");
    // 模拟一个最经典的神经网络全连接层： Y = X @ W^T + b
    // Batch Size = 2, In Features = 3, Out Features = 2
    auto X_linear = torch::ones<float>({2, 3}, true); // 输入数据 [2, 3]
    auto W = torch::ones<float>({2, 3}, true);        // 权重矩阵 [2, 3]
    W.fill_(2.0f);
    auto bias = torch::ones<float>({2}, true);        // 偏置向量 [2]
    bias.fill_(3.0f);

    // 1. W 的转置 W^T，形状变为 [3, 2]
    auto W_t = W.transpose(0, 1);
    
    // 2. X @ W^T，结果形状为 [2, 2]
    // X全是1，W^T全是2，矩阵乘法: 1*2 + 1*2 + 1*2 = 6
    auto M_linear = torch::matmul(X_linear, W_t); 
    
    // 3. 加上偏置 b (带广播: [2] -> [2, 2])
    // 结果: 6 + 3 = 9
    auto Y_linear = M_linear + bias; 
    
    // 4. 算个 Loss: 所有元素的和
    // 实际上就是 L = Y.sum()
    // 为了简单，我们手动乘以一个标量来模拟梯度回传
    auto L_linear = Y_linear * 1.0f;
    
    cout << "Y_linear (Expected: 9.0):\n" << Y_linear << "\n";
    L_linear.backward();

    // 手算梯度：
    // dL/dY = 1.0
    // Y = M + b  => dY/db = 1 (广播需 sum), dY/dM = 1
    // M = X @ W^T => dM/dW^T = X^T, dM/dX = W
    
    // bias.grad: [2, 2] 的 1 沿第 0 维 sum -> [2] 的 2.0
    cout << "bias.grad (Expected: 2.0):\n" << bias.grad() << "\n";
    
    // X.grad = dM * (W^T)^T = 1.0 @ W = [2, 2] 的 1 @ [2, 3] 的 2
    // => 1*2 + 1*2 = 4.0
    cout << "X_linear.grad (Expected: 4.0):\n" << X_linear.grad() << "\n";

    // W_t.grad = X^T @ dM = [3, 2] 的 1 @ [2, 2] 的 1 = [3, 2] 的 2.0
    // W.grad 应该是 W_t.grad 的转置，也就是 [2, 3] 的 2.0
    // 但是这里注意！由于你的 transpose 目前只是返回一个 view，
    // 当梯度回传给 W_t 时，它会传给 W_t 背后的那个真实的 W 节点！
    cout << "W.grad (Expected: 2.0 or view dependent):\n" << W.grad() << "\n";

    print_separator("Test 5: 疯狂的视图与聚合算子 (Unary Ops Backward)");
    // 这个测试旨在把你刚刚写的所有视图和聚合算子串起来
    auto V = torch::ones<float>({2, 3}, true); // V: [2, 3], 全是 1.0
    V.fill_(2.0f); // V: 全是 2.0
    
    // 1. Reshape: [2, 3] -> [6]
    auto V_flat = V.reshape({6});
    
    // 2. Unsqueeze: [6] -> [1, 6]
    auto V_unsq = V_flat.unsqueeze(0);
    
    // 3. Expand: [1, 6] -> [3, 6]
    auto V_exp = V_unsq.expand({3, 6});
    
    // 4. Sum (keep_dim = false): [3, 6] 沿着 dim 0 累加 -> [6]
    // 此时 V_exp 的每一列都是 3 个相同的 2.0，加起来应该是 6.0
    auto V_sum = V_exp.sum(0, false);
    
    // 5. 算个 Loss: [6] 所有的数加起来
    auto L_unary = V_sum.sum(0, false);
    
    cout << "L_unary (Expected: 36.0): " << L_unary.data_ptr()[0] << "\n";
    L_unary.backward();

    // 梯度回传手算推导：
    // dL/dV_sum: [6] 的全是 1.0
    // dL/dV_exp (SumBackward): 把 [6] 的 1.0 撑回到 [3, 6]，全是 1.0
    // dL/dV_unsq (ExpandBackward): 沿着 dim 0 把 [3, 6] 的 1.0 累加起来，变成 [1, 6] 的 3.0
    // dL/dV_flat (UnsqueezeBackward): 把 [1, 6] 的 3.0 挤成 [6] 的 3.0
    // dL/dV (ReshapeBackward): 把 [6] 的 3.0 变回 [2, 3] 的 3.0
    // 所以最终 V.grad 应该是一个 [2, 3] 的矩阵，里面全是 3.0！
    cout << "V.grad (Expected: 3.0 in shape [2, 3]):\n" << V.grad() << "\n";

    return 0;
}