#include "nn/data/datasets.h"
#include "nn/tensor.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int reverse_int(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float convert_to_float(unsigned char px) { return (float)px / 255.0f; }

std::vector<std::vector<std::vector<float>>> read_mnist(std::string path)
{
    std::ifstream file(path);
    std::vector<std::vector<std::vector<float>>> dataset;
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        // mnist files are big endian so need to reverse int
        magic_number = reverse_int(magic_number);
        if (magic_number != 2051)
        {
            throw std::runtime_error("Invalid MNIST image file!");
        }
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        for (int i = 0; i < number_of_images; ++i)
        {
            std::vector<std::vector<float>> image;
            for (int r = 0; r < n_rows; ++r)
            {
                std::vector<float> row;
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char *)&temp, sizeof(temp));
                    row.push_back(convert_to_float(temp));
                }
                image.push_back(row);
            }
            dataset.push_back(image);
        }
    }
    return dataset;
}

std::vector<int> read_mnist_labels(std::string path)
{
    std::ifstream file(path);
    std::vector<int> labels;
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_items = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        if (magic_number != 2049)
        {
            throw std::runtime_error("Invalid MNIST label file!");
        }
        file.read((char *)&number_of_items, sizeof(number_of_items));
        // mnist files are big endian so need to reverse int
        number_of_items = reverse_int(number_of_items);
        for (int i = 0; i < number_of_items; ++i)
        {
            unsigned char label = 0;
            file.read((char *)&label, sizeof(label));
            labels.push_back(label);
        }
    }
    return labels;
}

MNIST::MNIST(std::string data_path, std::string labels_path)
{
    _images = read_mnist(data_path);
    _labels = read_mnist_labels(labels_path);
}

std::pair<int, std::shared_ptr<Tensor>> MNIST::get_item(int index)
{
    return std::make_pair(_labels[index], std::make_shared<Tensor>(_images[index]));
}

int MNIST::get_length() { return _images.size(); }

std::string MNIST::label_to_class(int label) { return classes[label]; }

FashionMNIST::FashionMNIST(std::string data_path, std::string labels_path)
{
    _images = read_mnist(data_path);
    _labels = read_mnist_labels(labels_path);
}

std::pair<int, std::shared_ptr<Tensor>> FashionMNIST::get_item(int index)
{
    return std::make_pair(_labels[index], std::make_shared<Tensor>(_images[index]));
}

int FashionMNIST::get_length() { return _images.size(); }

std::string FashionMNIST::label_to_class(int label) { return _classes[label]; }

void visualize_image(std::shared_ptr<Tensor> image)
{
    for (int i = 0; i < image->shape()[0]; ++i)
    {
        for (int j = 0; j < image->shape()[1]; ++j)
        {
            float px = (*image)(i, j);
            std::cout << (px > 0.75   ? '@'
                          : px > 0.5  ? '#'
                          : px > 0.25 ? '+'
                          : px > 0.1  ? '.'
                                      : ' ');
        }
        std::cout << '\n';
    }
}