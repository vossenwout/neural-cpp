#pragma once
#include "nn/tensor.h"
#include <string>
#include <unordered_map>

void save(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict,
          const std::string &filename);

std::unordered_map<std::string, std::shared_ptr<Tensor>> load(const std::string &filename);
