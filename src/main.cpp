#include "nn/modules/flatten.h"
#include "nn/modules/linear.h"
#include "nn/modules/module.h"
#include "nn/modules/relu.h"
#include "nn/serialization.h"
#include "nn/tensor.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

class NeuralNetwork : public Module
{
private:
    // layers
    std::shared_ptr<Flatten> _flatten = std::make_shared<Flatten>();
    std::shared_ptr<Linear> _linear_1 = std::make_shared<Linear>(28 * 28, 512);
    std::shared_ptr<Linear> _linear_2 = std::make_shared<Linear>(512, 512);
    std::shared_ptr<Linear> _linear_3 = std::make_shared<Linear>(512, 10);
    // activation
    std::shared_ptr<Relu> _relu = std::make_shared<Relu>();

public:
    NeuralNetwork()
    {
        register_module("linear_1", _linear_1);
        register_module("linear_2", _linear_2);
        register_module("linear_3", _linear_3);
    }
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input)
    {
        std::shared_ptr<Tensor> flat = (*_flatten)(input);
        std::shared_ptr<Tensor> linear_1 = (*_linear_1)(flat);
        std::shared_ptr<Tensor> relu_1 = (*_relu)(linear_1);
        std::shared_ptr<Tensor> linear_2 = (*_linear_2)(relu_1);
        std::shared_ptr<Tensor> relu_2 = (*_relu)(linear_2);
        std::shared_ptr<Tensor> linear_3 = (*_linear_3)(relu_2);
        return linear_3;
    }
};

int main()
{
    NeuralNetwork model;

    // random input
    std::vector<std::vector<float>> input_data(28, std::vector<float>(28));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &row : input_data)
    {
        for (auto &val : row)
        {
            val = dist(rng);
        }
    }
    // create input tensor
    std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>(input_data);

    // forward pass
    std::shared_ptr<Tensor> output = model(input_tensor);

    std::cout << (*output) << std::endl;

    return 0;
}