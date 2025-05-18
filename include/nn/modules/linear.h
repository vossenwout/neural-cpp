#pragma once
#include "module.h"
#include "nn/tensor.h"
#include <memory>

class Linear : public Module
{
private:
    std::shared_ptr<Tensor> _weight;
    std::shared_ptr<Tensor> _bias;
    std::size_t _in_features;
    std::size_t _out_features;
    std::size_t _seed;

public:
    Linear(std::size_t in_features, std::size_t out_features, std::size_t seed = 7);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    void reset_parameters();
};