#include "nn/tensor.h"
#include <iostream>
#include <string>
#include <vector>

Tensor::Tensor(float data, bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _data{data}, _shape{}, _stride{}, _requires_grad(requires_grad), _gradfn(gradfn),
      _parents(parents)
{
    if (_requires_grad)
    {
        zero_grad();
    }
}

Tensor::Tensor(std::vector<float> data, bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _data(data), _shape{data.size()}, _stride{1}, _requires_grad(requires_grad), _gradfn(gradfn),
      _parents(parents)
{
    if (_requires_grad)
    {
        zero_grad();
    }
}

Tensor::Tensor(std::vector<std::vector<float>> data, bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1},
      _requires_grad(requires_grad), _gradfn(gradfn), _parents(parents)
{
    // check if dimensions match
    std::size_t n_expected_columns = data[0].size();
    for (std::size_t i = 0; i < data.size(); i++)
    {
        if (data[i].size() != n_expected_columns)
        {
            throw std::invalid_argument("Dimensions are inconsistent.");
        }
    }
    // store in row major format like pytorch and numpy
    for (std::size_t i = 0; i < data.size(); i++)
    {
        for (std::size_t j = 0; j < data[i].size(); j++)
        {
            _data.push_back(data[i][j]);
        }
    }
    if (_requires_grad)
    {
        zero_grad();
    }
}

const float &Tensor::item() const
{
    // works only with scalars and 1d tensors (like in pytorch)
    if (_data.size() == 1)
    {
        return _data[0];
    }
    else
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

float &Tensor::item()
{
    // works only with scalars and 1d tensors (like in pytorch)
    if (_data.size() == 1)
    {
        return _data[0];
    }
    else
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

const float &Tensor::operator()(std::size_t i) const
{
    if (_shape.size() == 0)
    {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (_shape.size() == 1)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Index " + std::to_string(i) +
                                        " is out of bounds for array of size " +
                                        std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}

float &Tensor::operator()(std::size_t i)
{
    if (_shape.size() == 0)
    {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (_shape.size() == 1)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Index " + std::to_string(i) +
                                        " is out of bounds for array of size " +
                                        std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const
{
    if (_shape.size() == 2)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Row index " + std::to_string(i) +
                                        " is out of bounds for tensor with " +
                                        std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1])
        {
            throw std::invalid_argument("Column index " + std::to_string(j) +
                                        " is out of bounds for tensor with " +
                                        std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensors");
}

float &Tensor::operator()(std::size_t i, std::size_t j)
{
    if (_shape.size() == 2)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Row index " + std::to_string(i) +
                                        " is out of bounds for tensor with " +
                                        std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1])
        {
            throw std::invalid_argument("Column index " + std::to_string(j) +
                                        " is out of bounds for tensor with " +
                                        std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensors");
}

const std::vector<std::size_t> &Tensor::shape() const { return _shape; }

const std::vector<std::size_t> &Tensor::stride() const { return _stride; }

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other)
{
    // scalar + scalar
    if (_shape.size() == 0 && other->shape().size() == 0)
    {
        float result = item() + other->item();
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                // propagate parent gradient
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // scalar + 1D
    if (_shape.size() == 0 && other->shape().size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < other->shape()[0]; i++)
        {
            result.push_back(item() + ((*other)(i)));
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                // broadcast in forward == sum in backward
                float grad_self = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++)
                {
                    grad_self += grad_output[i];
                }
                self->add_to_grad({grad_self});
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // scalar + 2D
    if (_shape.size() == 0 && other->shape().size() == 2)
    {
        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < other->shape()[0]; i++)
        {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < other->shape()[1]; j++)
            {
                result_i.push_back(item() + (*other)(i, j));
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                // broadcast forward == sum in backward
                float grad_self = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++)
                {
                    grad_self += grad_output[i];
                }
                self->add_to_grad({grad_self});
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D + scalar
    if (_shape.size() == 1 && other->shape().size() == 0)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            result.push_back(operator()(i) + other->item());
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                self->add_to_grad(grad_output);
                // broadcast in forward == sum in backward
                float grad_other = 0.0f;
                for (std::size_t i; i < grad_output.size(); i++)
                {
                    grad_other += grad_output[i];
                }
                other->add_to_grad({grad_other});
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D + scalar
    if (_shape.size() == 2 && other->shape().size() == 0)
    {
        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < shape()[1]; j++)
            {
                result_i.push_back(operator()(i, j) + other->item());
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                // propagate child grad
                self->add_to_grad(grad_output);
                // broadcast in forward == sum in backward
                float grad_other = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++)
                {
                    grad_other += grad_output[i];
                }
                other->add_to_grad({grad_other});
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D + 1D
    if (_shape[0] != other->shape()[0])
    {
        throw std::invalid_argument("First dimensions are not equal.");
    }
    if (_shape.size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            result.push_back(operator()(i) + (*other)(i));
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                // propagate child gradients
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D + 2D
    else
    {
        if (shape()[1] != other->shape()[1])
        {
            throw std::invalid_argument("Second dimensions are not equal.");
        }
        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < shape()[1]; j++)
            {
                result_i.push_back(operator()(i, j) + (*other)(i, j));
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                // propagate child gradient
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other)
{
    if (_shape.size() == 0 || other->shape().size() == 0)
    {
        throw std::invalid_argument("Both arguments needs to be at least 1D for matmul.");
    }
    if (_shape[_shape.size() - 1] != other->shape()[0])
    {
        throw std::invalid_argument(
            "Last dimension of first tensor doesn't have same size as first dimension of second.");
    }
    // 1D x 1D -> float
    if (_shape.size() == 1 && other->shape().size() == 1)
    {
        float result = 0.0f;
        for (std::size_t i = 0; i < _shape[0]; i++)
        {
            result += operator()(i) * (*other)(i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                std::vector<float> grad_self;
                std::vector<float> grad_other;
                for (std::size_t i = 0; i < self->numel(); i++)
                {
                    // output gradients is a scalar and gets propagated back to local gradients
                    grad_self.push_back((*other)(i)*grad_output[0]);
                    grad_other.push_back((*self)(i)*grad_output[0]);
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D x 1D -> 1D
    else if (_shape.size() == 2 && other->shape().size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < _shape[0]; i++)
        {
            float result_i = 0.0f;
            for (std::size_t j = 0; j < _shape[1]; j++)
            {
                result_i += operator()(i, j) * (*other)(j);
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                std::vector<float> grad_self;
                // iterate over row major order
                for (std::size_t i = 0; i < self->shape()[0]; i++)
                {
                    for (std::size_t j = 0; j < self->shape()[1]; j++)
                    {
                        // everything from the i_th row contributes to the i_th child gradient
                        grad_self.push_back((*other)(j)*grad_output[i]);
                    }
                }
                std::vector<float> grad_other;
                for (std::size_t i = 0; i < other->shape()[0]; i++)
                {
                    // iterate through rows for the i'th column
                    float grad_other_i = 0.0f;
                    for (std::size_t j; j < self->shape()[0]; j++)
                    {
                        // j_th derivated of child propagates to j_th row
                        grad_other_i += (*self)(j, i) * grad_output[j];
                    }
                    grad_other.push_back(grad_other_i);
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 1D x 2D -> 1D
    else if (_shape.size() == 1 && other->shape().size() == 2)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < other->shape()[1]; i++)
        {
            float result_i = 0.0f;
            for (std::size_t j = 0; j < other->shape()[0]; j++)
            {
                result_i += operator()(j) * (*other)(j, i);
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                std::vector<float> grad_self;
                // specify row
                for (std::size_t i = 0; i < self->shape()[0]; i++)
                {
                    // iterate over columns
                    float grad_self_i = 0.0f;
                    for (std::size_t j = 0; j < other->shape()[1]; j++)
                    {
                        // j_th derivate of chuld propagates to j_th column
                        grad_self_i += (*other)(i, j) * grad_output[j];
                    }
                    grad_self.push_back(grad_self_i);
                }
                std::vector<float> grad_other;
                // iterate in row major order
                for (std::size_t i = 0; i < other->shape()[0]; i++)
                {
                    for (std::size_t j = 0; j < other->shape()[1]; j++)
                    {
                        // everything from the j_th column contributes to the j_th child gradient
                        grad_other.push_back((*self)(i)*grad_output[j]);
                    }
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D x 2D
    else
    {
        if (other->shape().size() < 2)
        {
            throw std::invalid_argument(
                "Expected second tensor to have at least 2 dimensions for this operation");
        }
        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < other->shape()[1]; j++)
            {
                float result_i_j = 0.0f;
                for (std::size_t k = 0; k < shape()[1]; k++)
                {
                    result_i_j += operator()(i, k) * (*other)(k, j);
                }
                result_i.push_back(result_i_j);
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                std::vector<float> grad_self;
                // iterate over rows self
                for (std::size_t i = 0; i < self->shape()[0]; i++)
                {
                    // iterate over columns self
                    for (std::size_t j = 0; j < self->shape()[1]; j++)
                    {
                        float grad_self_i_j = 0.0f;
                        // j_th column of self gets sum of the j_th row of other
                        // iterate over row other
                        for (std::size_t k = 0; k < other->shape()[1]; k++)
                        {
                            // and propagates corresponding child_grad_i_k
                            // grad_output is also stored in row major order
                            grad_self_i_j +=
                                (*other)(j, k) * grad_output[i * other->shape()[1] + k];
                        }
                        grad_self.push_back(grad_self_i_j);
                    }
                }
                std::vector<float> grad_other;
                // iterate over rows other
                for (std::size_t i = 0; i < other->shape()[0]; i++)
                {
                    // iterate over columns other
                    for (std::size_t j = 0; j < other->shape()[1]; j++)
                    {
                        float grad_other_i_j = 0.0f;
                        // i_th row of other gets summed with ith column of self
                        for (std::size_t k = 0; k < self->shape()[0]; k++)
                        {
                            // and propagates corresponding child_grad_k_j
                            // grad_output is also stored in row major format
                            grad_other_i_j +=
                                (*self)(k, i) * grad_output[k * other->shape()[1] + j];
                        }
                        grad_other.push_back(grad_other_i_j);
                    }
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
}

void Tensor::backward()
{
    // public interface to set initial gradient
    if (!_requires_grad)
    {
        throw std::runtime_error("Element does not require grad.");
    }
    if (_shape.size() != 0)
    {
        throw std::runtime_error("Grad can only be calculated for scalar outputs.");
    }
    _reset_graph_visit();
    _grad = {1.0f};
    _backward();
}

void Tensor::_backward()
{
    if (!_requires_grad)
    {
        return;
    }
    if (_visited)
    {
        return;
    }
    _visited = true;
    if (_gradfn)
    {
        _gradfn(_grad);
    }
    for (std::size_t i = 0; i < _parents.size(); i++)
    {
        _parents[i]->_backward();
    }
}

const bool &Tensor::requires_grad() const { return _requires_grad; };

const std::vector<float> &Tensor::grad() const { return _grad; }

void Tensor::add_to_grad(const std::vector<float> &grad_update)
{
    if (!_requires_grad)
    {
        return;
    }
    if (_grad.size() != grad_update.size())
    {
        throw std::runtime_error("Gradient shape mismatch during accumulation.");
    }
    for (std::size_t i = 0; i < _grad.size(); i++)
    {
        _grad[i] += grad_update[i];
    }
}

void Tensor::zero_grad() { _grad = std::vector<float>(_data.size(), 0.0f); }

void Tensor::_reset_graph_visit()
{
    // Recursively reset '_visited' in entire graph.
    if (!_visited)
    {
        return;
    }
    _visited = false;
    for (std::size_t i = 0; i < _parents.size(); i++)
    {
        _parents[i]->_reset_graph_visit();
    }
}

std::size_t Tensor::numel() const { return _data.size(); }

std::vector<float> &Tensor::data() { return _data; }

std::size_t Tensor::argmax() const
{
    return std::distance(_data.begin(), std::max_element(_data.begin(), _data.end()));
}

std::ostream &operator<<(std::ostream &os, const Tensor &obj)
{
    std::string string_repr = "[";
    if (obj.shape().size() == 0)
    {
        os << obj.item();
        return os;
    }
    else if (obj.shape().size() == 1)
    {
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            string_repr += std::to_string(obj(i));
            if (i != obj.shape()[0] - 1)
            {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    else
    {
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            string_repr += "[";
            for (std::size_t j = 0; j < obj.shape()[1]; j++)
            {
                string_repr += std::to_string(obj(i, j));
                if (j != obj.shape()[1] - 1)
                {
                    string_repr += ", ";
                }
            }
            string_repr += "]";
            if (i != obj.shape()[0] - 1)
            {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    os << string_repr;
    return os;
}
