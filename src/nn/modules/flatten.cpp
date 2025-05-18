#include "nn/modules/flatten.h"
#include "nn/tensor.h"
#include <functional>
#include <memory>

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input)
{
    std::vector<float> result;
    if (input->shape().size() == 0)
    {
        result.push_back(input->item());
    }
    else if (input->shape().size() == 1)
    {
        for (std::size_t i = 0; i < input->shape()[0]; i++)
        {
            result.push_back((*input)(i));
        }
    }
    else
    {
        for (std::size_t i = 0; i < input->shape()[0]; i++)
        {
            for (std::size_t j = 0; j < input->shape()[1]; j++)
            {
                result.push_back((*input)(i, j));
            }
        }
    }
    if (input->requires_grad())
    {
        std::vector<std::shared_ptr<Tensor>> parents{input};

        std::function<void(const std::vector<float> &)> gradfn =
            [input](const std::vector<float> &grad_output)
        {
            std::vector<float> grad_input;
            // our grad is always stored in row major order
            // so just propagate grad of the output
            input->add_to_grad(grad_output);
        };
        return std::make_shared<Tensor>(result, true, gradfn, parents);
    }
    return std::make_shared<Tensor>(result);
}