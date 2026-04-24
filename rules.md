# Project Rules

## Dimension Indexing Rule
- Index `0` always represents the **highest dimension** (the outermost dimension).
- Index `ndim - 1` always represents the **lowest dimension** (the innermost dimension, or contiguous dimension).

## C++ Features & Design Patterns in nn::Module

- **Polymorphism & Virtual Destructor (多态与虚析构)**: 
  The base class must define `virtual ~Module() = default;`. This ensures that when a derived class (e.g., `Linear`) is destroyed via a base class pointer, the derived destructor is correctly invoked, preventing memory leaks.
- **Smart Pointers for Object Slicing Prevention (智能指针防切片)**: 
  Child modules are stored using `std::shared_ptr<Module<real>>`. This is mandatory because storing objects directly by value would cause "Object Slicing" (对象切片), losing the derived class's specific data and overridden methods. It also automates lifecycle management.
- **Member Templates (成员函数模板)**: 
  The method `template <typename ModuleType> std::shared_ptr<ModuleType> register_module(...)` leverages C++ templates to accept any derived module type and return that exact derived type. This mimics libtorch's capability for elegant chain assignments in constructors.
- **C++17 Structured Binding (C++17 结构化绑定)**: 
  Used extensively for elegant map iteration, e.g., `for (const auto& [name, param] : parameters_)`. This unpacks the `std::pair` directly into readable variables, eliminating the need for `.first` and `.second`.
- **Deterministic Iteration via std::map (基于红黑树的确定性迭代)**: 
  Instead of `std::unordered_map` (Hash Table), we strictly use `std::map` (Red-Black Tree) for `parameters_` and `modules_`. This guarantees that iteration order is strictly deterministic (alphabetical). Determinism is critical in Deep Learning for consistent weight saving/loading and aligned optimizer updates.

  ### Deep Dive: std::map vs std::unordered_map in Deep Learning
- **The Problem with Hash Tables**: `std::unordered_map` uses hashing. Its iteration order depends on the hash function, bucket count, and insertion history, which means the order is **non-deterministic** across different runs, compilers, or platforms.
- **Why Determinism Matters**: In a neural network, an Optimizer (like SGD) flattens all parameters into a list to update them. If the order of parameters changes randomly, the optimizer might apply the wrong momentum/gradients to the wrong weights. Furthermore, when serializing the model to disk (saving a `state_dict`), a deterministic order ensures the weights are correctly aligned when loaded back.
- **The Solution**: `std::map` is backed by a Red-Black Tree. It guarantees that elements are always iterated in strict alphabetical order of their string keys (e.g., "bias" will always come before "weight"). While it has a slightly slower lookup time ($O(\log N)$ vs $O(1)$), network initialization and parameter registration only happen once during construction, making this a perfect trade-off for safety.

### Deep Dive: Why 'protected' instead of 'private' or 'public'?
- **Encapsulation (Hiding from Outside)**: We do not want external users (like the training loop) to arbitrarily insert or delete parameters. They should only interact via the public API (`register_parameter`, `parameters()`, `zero_grad()`). Thus, `public` is strictly forbidden for `parameters_` and `modules_`.
- **Inheritance (Visibility to Children)**: If we used `private`, derived classes (like `nn::Linear` or custom `MyModel`) would have absolutely no access to these containers or the `is_training_` flag. While `register_parameter` handles insertion, a derived class might legitimately need to check if a specific sub-module exists or read the `is_training_` flag to alter its `forward()` behavior (e.g., Dropout behaves differently in train vs eval mode).
- **The Balance**: `protected` perfectly models the "Base-Derived" relationship. It keeps the internal state safe from the outside world but maintains a trusted environment within the `nn::Module` class hierarchy.