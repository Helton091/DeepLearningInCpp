#include"Torch.hpp"
#include<iostream>
using std::cout,std::endl;
using namespace torch;

void print_separator(const std::string& title) {
    cout << "\n========================================\n"
         << "   " << title << "\n"
         << "========================================\n";
}

int main(){
    print_separator("1. Tensor Generation");
    auto A = torch::arange<double>(1.0, 7.0).reshape({2, 3});
    auto B = torch::ones<double>({2, 3}) * 2.0;
    auto rand_t = torch::randn<double>({2, 2});
    cout << "A (arange):\n" << A << "\n";
    cout << "B (ones * 2):\n" << B << "\n";
    cout << "rand_t:\n" << rand_t << "\n";

    print_separator("2. Basic Operations & Broadcasting");
    auto C = A + B;
    cout << "A + B:\n" << C << "\n";
    
    // Broadcasting: [2, 3] * [3] -> [2, 3]
    auto vec = torch::arange<double>(1.0, 4.0);
    cout << "vec (shape [3]):\n" << vec << "\n";
    auto D = A * vec;
    cout << "A * vec (Broadcasting):\n" << D << "\n";

    print_separator("3. View Operations (Zero-copy)");
    auto A_t = A.transpose(0, 1);
    cout << "A.transpose(0, 1) -> shape [3, 2]:\n" << A_t << "\n";
    cout << "Is A_t contiguous? " << (A_t.is_contiguous() ? "True" : "False") << "\n";

    auto A_perm = A.unsqueeze(0).permute({0, 2, 1});
    cout << "A.unsqueeze(0).permute({0, 2, 1}) -> shape [1, 3, 2]:\n" << A_perm << "\n";

    print_separator("4. Operations on Non-contiguous Tensors");
    // [3, 2] + [3, 2] where A_t is non-contiguous
    auto E = A_t + torch::ones<double>({3, 2});
    cout << "A_t + ones({3, 2}):\n" << E << "\n";

    print_separator("5. Squeeze & Unsqueeze");
    auto squeezed = A_perm.squeeze(0);
    cout << "Squeeze dim 0 -> shape [3, 2]:\n" << squeezed << "\n";
    
    auto unsqueezed = squeezed.unsqueeze(-1);
    cout << "Unsqueeze at -1 -> shape [3, 2, 1]:\n" << unsqueezed << "\n";

    print_separator("6. Contiguous Memory Layout");
    auto contig_A_t = A_t.contiguous();
    cout << "contig_A_t:\n" << contig_A_t << "\n";
    cout << "Is contig_A_t contiguous? " << (contig_A_t.is_contiguous() ? "True" : "False") << "\n";
    
    return 0;
}