#include "nn/tensor.h"
#include <iostream>
#include <vector>
int main()
{
    Tensor test_tensor({{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}});

    std::cout << test_tensor << std::endl;

    std::cout << test_tensor(0, 2) << std::endl;

    return 0;
}