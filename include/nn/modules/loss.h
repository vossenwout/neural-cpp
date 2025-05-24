#pragma once
#include "module.h"
#include "nn/tensor.h"
#include <memory>

class Loss : public Module
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target);
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input, std::size_t target);
};

class NLLLoss : public Loss
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target) override;
};

class CrossEntropyLoss : public Loss
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target) override;
};