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
    print_separator("1. 1D @ 1D (Dot Product)");
    auto v1 = torch::arange<float>(1.0, 4.0); // [1, 2, 3]
    auto v2 = torch::arange<float>(2.0, 5.0); // [2, 3, 4]
    cout << "v1: " << v1 << "\n";
    cout << "v2: " << v2 << "\n";
    cout << "v1 @ v2 = " << torch::matmul(v1, v2) << "\n"; 
    // Expected: 1*2 + 2*3 + 3*4 = 20

    print_separator("2. 2D @ 2D (Standard Matrix Multiplication)");
    auto A = torch::arange<float>(1.0, 7.0).reshape({2, 3}); 
    auto B = torch::arange<float>(1.0, 7.0).reshape({3, 2}); 
    cout << "A (2x3):\n" << A << "\n";
    cout << "B (3x2):\n" << B << "\n";
    cout << "A @ B:\n" << torch::matmul(A, B) << "\n";
    // Expected: [[22, 28], [49, 64]]

    print_separator("3. 2D @ 2D (Non-contiguous Tensors)");
    // Transposing creates non-contiguous views with different strides
    auto A_t = A.transpose(0, 1); // Shape: [3, 2]
    auto B_t = B.transpose(0, 1); // Shape: [2, 3]
    cout << "A_t (3x2, non-contiguous):\n" << A_t << "\n";
    cout << "B_t (2x3, non-contiguous):\n" << B_t << "\n";
    cout << "A_t @ B_t:\n" << torch::matmul(A_t, B_t) << "\n";

    print_separator("4. 1D @ 2D and 2D @ 1D (Unsqueeze & Squeeze)");
    cout << "v1 @ B (1D @ 2D):\n" << torch::matmul(v1, B) << "\n"; // [1, 2, 3] @ 3x2
    cout << "A @ v1 (2D @ 1D):\n" << torch::matmul(A, v1) << "\n"; // 2x3 @ [1, 2, 3]

    print_separator("5. Batched Matmul with Broadcasting (The Boss Fight)");
    // A: shape [1, 3, 2, 4], filled with 1.0
    // B: shape [3, 1, 4, 2], filled with 2.0
    auto batch_A = torch::ones<float>({1, 3, 2, 4});
    auto batch_B = torch::ones<float>({3, 1, 4, 2}) * 2.0f;
    
    auto batch_C = torch::matmul(batch_A, batch_B);
    
    cout << "Shape of batch_A: [1, 3, 2, 4]\n";
    cout << "Shape of batch_B: [3, 1, 4, 2]\n";
    cout << "Shape of batch_C: [";
    for (size_t i = 0; i < batch_C.shape().size(); ++i) {
        cout << batch_C.shape()[i] << (i == batch_C.shape().size() - 1 ? "" : ", ");
    }
    cout << "]\n";
    // Expected Shape: [3, 3, 2, 2]
    
    // Since 1.0 * 2.0 is accumulated 4 times (K=4), every element should be 8.0
    cout << "Is the first element of batch_C 8.0? " 
         << (batch_C.data_ptr()[0] == 8.0f ? "Yes!" : "No!") << "\n";

    return 0;
}