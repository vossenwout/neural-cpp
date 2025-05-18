#include "nn/serialization.h"
#include "nn/tensor.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

const int MAGIC_NUMBER = 777;

void save(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict,
          const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&MAGIC_NUMBER), sizeof(int));
    for (const auto &[weight_name, weight] : state_dict)
    {
        size_t name_len = weight_name.size();
        file.write(reinterpret_cast<const char *>(&name_len), sizeof(size_t));
        file.write(weight_name.data(), name_len);

        size_t shape_length = weight->shape().size();
        file.write(reinterpret_cast<const char *>(&shape_length), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(weight->shape().data()),
                   shape_length * sizeof(size_t));

        size_t data_length = weight->numel();
        file.write(reinterpret_cast<const char *>(&data_length), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(weight->data().data()),
                   data_length * sizeof(float));
    }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> load(const std::string &filename)
{
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open " + filename);
    }

    int magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(int));
    if (magic != MAGIC_NUMBER)
    {
        throw std::runtime_error("Bad fle format: wrong magic number");
    }

    while (file.peek() != EOF)
    {
        size_t name_len = 0;
        if (!file.read(reinterpret_cast<char *>(&name_len), sizeof(size_t)))
            break;

        std::string weight_name(name_len, '\0');
        file.read(weight_name.data(), name_len);

        size_t shape_length = 0;
        file.read(reinterpret_cast<char *>(&shape_length), sizeof(size_t));

        std::vector<size_t> shape(shape_length);
        file.read(reinterpret_cast<char *>(shape.data()), shape_length * sizeof(size_t));

        size_t data_length = 0;
        file.read(reinterpret_cast<char *>(&data_length), sizeof(size_t));

        std::vector<float> raw(data_length);
        file.read(reinterpret_cast<char *>(raw.data()), data_length * sizeof(float));

        std::shared_ptr<Tensor> tensor;

        if (shape_length == 0)
        {
            tensor = std::make_shared<Tensor>(raw[0]);
        }
        else if (shape_length == 1)
        {
            tensor = std::make_shared<Tensor>(raw);
        }
        else if (shape_length == 2)
        {
            std::vector<std::vector<float>> data_2d(shape[0], std::vector<float>(shape[1]));
            for (size_t i = 0; i < shape[0]; i++)
            {
                for (size_t j = 0; j < shape[1]; j++)
                {
                    data_2d[i][j] = raw[i * shape[1] + j];
                }
            }
            tensor = std::make_shared<Tensor>(data_2d);
        }
        else
        {
            throw std::runtime_error("Unsupported tensor dimensionality: " +
                                     std::to_string(shape_length));
        }

        state_dict[weight_name] = tensor;
    }

    return state_dict;
}