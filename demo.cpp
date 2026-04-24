#include "Torch.hpp"
#include "Torch/nn/Sequential.hpp"
#include "Torch/nn/Linear.hpp"
#include "Torch/nn/Functional/MseLoss.hpp"
#include "Torch/Optim/SGD.hpp"
#include <iostream>

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;
using std::cout;

void print_separator(const std::string& title) {
    cout << "\n========================================\n"
         << "   " << title << "\n"
         << "========================================\n";
}

int main() {
    print_separator("C++ Deep Learning Engine - Linear Regression Test");
    cout << "Target Function: y = 2x + 1\n\n";

    // 1. 准备训练数据 (Training Data)
    // X 形状为 [BatchSize=4, in_features=1]
    // Y 形状为 [BatchSize=4, out_features=1]
    int batch_size = 4;
    Tensor<float> X({batch_size, 1}, false); // 数据不需要计算梯度
    Tensor<float> Y_true({batch_size, 1}, false);
    
    // 手动填充数据: X = [1, 2, 3, 4]^T, Y_true = 2X + 1 = [3, 5, 7, 9]^T
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y_data[] = {3.0f, 5.0f, 7.0f, 9.0f};
    for (int i = 0; i < batch_size; ++i) {
        X.data_ptr()[i] = x_data[i];
        Y_true.data_ptr()[i] = y_data[i];
    }

    // 2. 构建模型 (Build Model)
    // 我们用 Sequential 包装一个 Linear 层，输入特征数=1，输出特征数=1
    auto model = std::make_shared<Sequential<float>>(
        std::make_shared<Linear<float>>(1, 1, true)
    );

    // 打印初始权重（应该是由 randn 随机初始化的）
    cout << "Initial Parameters:\n";
    auto params = model->named_parameters();
    for (const auto& kv : params) {
        cout << kv.first << ":\n" << kv.second << "\n";
    }

    // 3. 构建优化器 (Build Optimizer)
    // 学习率设置得稍微小一点 (0.01)，因为我们的 MSE 是求和而不是求平均，梯度会比较大
    SGD<float> optimizer(model->parameters(), 0.01f);

    // 4. 训练循环 (Training Loop)
    int epochs = 100;
    cout << "\nStarting Training...\n";
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Step 1: 前向传播
        Tensor<float> Y_pred = model->forward(X);
        
        // Step 2: 计算损失 (MSE Loss)
        Tensor<float> loss = functional::mse_loss(Y_pred, Y_true);
        
        // 每 10 个 Epoch 打印一次状态
        if (epoch % 10 == 0 || epoch == 1) {
            cout << "Epoch [" << epoch << "/" << epochs << "] Loss: " << loss.data_ptr()[0] << "\n";
        }
        
        // Step 3: 清空梯度
        optimizer.zero_grad();
        
        // Step 4: 反向传播
        loss.backward();
        
        // Step 5: 更新权重
        optimizer.step();
    }

    print_separator("Training Finished");

    // 5. 验证结果 (Verification)
    cout << "Final Parameters (Expected: weight ~= 2.0, bias ~= 1.0):\n";
    auto final_params = model->named_parameters();
    for (const auto& kv : final_params) {
        cout << kv.first << ":\n" << kv.second << "\n";
    }

    // 测试一个新的输入: X_test = [10]
    Tensor<float> X_test({1, 1}, false);
    X_test.data_ptr()[0] = 10.0f;
    Tensor<float> Y_test = model->forward(X_test);
    cout << "\nTest Prediction for x=10 (Expected ~= 21.0):\n" << Y_test << "\n";

    return 0;
}