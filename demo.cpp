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
    print_separator("Level 1: 基础加法反向传播 (Basic Addition)");
    // A: [2, 2], requires_grad = true
    auto A = torch::ones<float>({2, 2}, true); 
    
    // B: [2, 2], requires_grad = true, 初始值为 2.0
    // 【修复】：使用 fill_ 就地赋值，防止 operator* 生成一个丢了 requires_grad 的新张量
    auto B = torch::ones<float>({2, 2}, true);
    B.fill_(2.0f); 

    auto C = A + B; // C 应该全为 3.0
    cout << "C (A + B) requires_grad: " << (C.requires_grad() ? "true" : "false") << "\n";
    
    C.backward(); // 启动你手搓的拓扑排序引擎！
    
    // A 和 B 的局部偏导都是 1，且传进来的 grad_output 默认是 1
    // 所以 A 和 B 的梯度应该全都是 1.0
    cout << "A.grad:\n" << A.grad() << "\n";
    cout << "B.grad:\n" << B.grad() << "\n";

    print_separator("Level 2: 带有广播机制的加法 (Broadcast Addition)");
    auto X = torch::ones<float>({2, 3}, true);      // 形状 [2, 3]
    
    // Y: 形状 [3]，一维向量，初始值为 5.0
    auto Y = torch::ones<float>({3}, true);
    Y.fill_(5.0f);

    auto Z = X + Y; // Z 形状 [2, 3]
    Z.backward();

    // X 的梯度形状应该是 [2, 3]，全为 1.0
    cout << "X.grad (should be [2, 3] of 1s):\n" << X.grad() << "\n";
    // Y 参与了广播，它的梯度必须经过 unbroadcast 累加！
    // 因为 Z 的行数为 2，所以 Y 的梯度应该是 [1.0 + 1.0, 1.0 + 1.0, 1.0 + 1.0] = [2.0, 2.0, 2.0]
    cout << "Y.grad (should be [3] of 2s, testing unbroadcast):\n" << Y.grad() << "\n";

    print_separator("Level 3: 终极考验 - 复杂 DAG 与残差连接 (ResNet Style)");
    // 这是对拓扑排序入度算法的最强考验！
    auto M = torch::ones<float>({2, 2}, true);
    M.fill_(10.0f);
    
    auto N = M + M; // N = 2M. 对 M 的偏导是 2
    auto L = N + M; // L = N + M = 3M. 对 M 的总偏导应该是 3
    
    L.backward();

    // 如果拓扑排序写错了，或者入度统计有 Bug（比如多次入队），这里的梯度就会错。
    // 正确答案：M 的梯度应该是全 3.0！
    cout << "M.grad (should be [2, 2] of 3s):\n" << M.grad() << "\n";

    return 0;
}