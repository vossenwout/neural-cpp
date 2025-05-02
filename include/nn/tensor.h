#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
class Tensor
{
private:
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride;

public:
    Tensor(float data);
    Tensor(std::vector<float> data);
    Tensor(std::vector<std::vector<float>> data);
    const float &item() const;
    float &item();
    const float &operator()(std::size_t i) const;
    float &operator()(std::size_t i);
    const float &operator()(std::size_t i, std::size_t j) const;
    float &operator()(std::size_t i, std::size_t j);
    const std::vector<std::size_t> &shape() const;
    const std::vector<std::size_t> &stride() const;
    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
};