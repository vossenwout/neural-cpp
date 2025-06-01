#pragma once
#include "nn/tensor.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Dataset
{
public:
    virtual std::pair<int, std::shared_ptr<Tensor>> get_item(int index) = 0;
    virtual int get_length() = 0;
};

class MNIST : public Dataset
{
private:
    std::vector<std::vector<std::vector<float>>> _images;
    std::vector<int> _labels;
    std::vector<std::string> classes = {"zero", "one", "two",   "three", "four",
                                        "five", "six", "seven", "eight", "nine"};

public:
    MNIST(std::string data_path, std::string labels_path);
    std::pair<int, std::shared_ptr<Tensor>> get_item(int index) override;
    int get_length() override;
    std::string label_to_class(int label);
};

class FashionMNIST : public Dataset
{
private:
    std::vector<std::vector<std::vector<float>>> _images;
    std::vector<int> _labels;
    std::vector<std::string> _classes = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
    };

public:
    FashionMNIST(std::string data_path, std::string labels_path);
    std::pair<int, std::shared_ptr<Tensor>> get_item(int index) override;
    int get_length() override;
    std::string label_to_class(int label);
};

void visualize_image(std::shared_ptr<Tensor> image);