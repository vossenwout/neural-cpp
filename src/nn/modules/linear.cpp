#include "nn/modules/linear.h"
#include "nn/tensor.h"
#include <memory>
#include <random>

Linear::Linear(std::size_t in_features, std::size_t out_features, std::size_t seed)
    : _in_features(in_features), _out_features(out_features),
      _weight(std::make_shared<Tensor>(
          std::vector<std::vector<float>>(in_features, std::vector<float>(out_features, 0.0f)),
          true)),
      _bias(std::make_shared<Tensor>(std::vector<float>(out_features, 0.0f), true)), _seed(seed)
{
    // register parameters
    register_parameter("weight", _weight);
    register_parameter("bias", _bias);
    // kaiming init
    reset_parameters();
}

void Linear::reset_parameters()
{
    // relu gain (same as pytorch)
    float gain = std::sqrt(2.0f);
    std::size_t fan_in = _in_features;
    float bound = gain * std::sqrt(3.0f / fan_in);
    std::mt19937 generator(_seed);

    for (std::size_t i = 0; i < _weight->shape()[0]; i++)
    {
        for (std::size_t j = 0; j < _weight->shape()[1]; j++)
        {
            (*_weight)(i, j) = std::uniform_real_distribution<float>(-bound, bound)(generator);
        }
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input)
{
    std::shared_ptr<Tensor> xW = (*input) * _weight;
    return (*xW) + _bias;
}