#include "nn/modules/flatten.h"
#include "nn/modules/linear.h"
#include "nn/modules/loss.h"
#include "nn/modules/relu.h"
#include "nn/modules/softmax.h"
#include "nn/serialization.h"
#include "nn/sgd.h"
#include "nn/tensor.h"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

TEST(TensorTest, Creation)
{
    // scalar
    Tensor tensor = Tensor(5.0);
    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({}));
    EXPECT_THROW(tensor(0), std::invalid_argument);
    EXPECT_EQ(tensor.item(), 5.0);

    // 1d
    std::vector<float> v = {1.0, 2.0, 3.0};
    Tensor tensor2 = Tensor(v);
    EXPECT_EQ(tensor2.shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ(tensor2(0), 1.0);
    EXPECT_EQ(tensor2(1), 2.0);
    EXPECT_EQ(tensor2(2), 3.0);
    EXPECT_THROW(tensor2(3), std::invalid_argument);
    EXPECT_THROW(tensor2.item(), std::runtime_error);

    // 2d
    std::vector<std::vector<float>> v_2 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    Tensor tensor3 = Tensor(v_2);
    EXPECT_EQ(tensor3.shape(), std::vector<std::size_t>({2, 3}));
    EXPECT_EQ(tensor3.stride(), std::vector<std::size_t>({3, 1}));
    EXPECT_EQ(tensor3(0, 0), 1.0);
    EXPECT_EQ(tensor3(0, 1), 2.0);
    EXPECT_EQ(tensor3(0, 2), 3.0);
    EXPECT_EQ(tensor3(1, 0), 4.0);
    EXPECT_EQ(tensor3(1, 1), 5.0);
    EXPECT_EQ(tensor3(1, 2), 6.0);
    EXPECT_THROW(tensor3(2, 0), std::invalid_argument);
    EXPECT_THROW(tensor3(0, 3), std::invalid_argument);
    EXPECT_THROW(tensor3.item(), std::runtime_error);
}

TEST(TensorTest, Addition)
{
    // scalar + scalar
    std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(2.0);
    std::shared_ptr<Tensor> tensor3 = (*tensor1) + tensor2;
    EXPECT_EQ(tensor3->item(), 3.0);

    // scalar + 1d
    std::shared_ptr<Tensor> tensor4 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor5 = std::make_shared<Tensor>(std::vector<float>({2.0, 3.0, 4.0}));
    std::shared_ptr<Tensor> tensor6 = (*tensor4) + tensor5;
    EXPECT_EQ(tensor6->shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*tensor6)(0), 3.0);
    EXPECT_EQ((*tensor6)(1), 4.0);
    EXPECT_EQ((*tensor6)(2), 5.0);

    // 1d + 1d
    std::shared_ptr<Tensor> tensor7 = std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 3.0}));
    std::shared_ptr<Tensor> tensor8 = std::make_shared<Tensor>(std::vector<float>({4.0, 5.0, 6.0}));
    std::shared_ptr<Tensor> tensor9 = (*tensor7) + tensor8;
    EXPECT_EQ(tensor9->shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*tensor9)(0), 5.0);
    EXPECT_EQ((*tensor9)(1), 7.0);
    EXPECT_EQ((*tensor9)(2), 9.0);

    // 1d + scalar
    std::shared_ptr<Tensor> tensor10 =
        std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 3.0}));
    std::shared_ptr<Tensor> tensor11 = std::make_shared<Tensor>(4.0);
    std::shared_ptr<Tensor> tensor12 = (*tensor10) + tensor11;
    EXPECT_EQ(tensor12->shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*tensor12)(0), 5.0);
    EXPECT_EQ((*tensor12)(1), 6.0);
    EXPECT_EQ((*tensor12)(2), 7.0);

    // 2d + 2d
    std::shared_ptr<Tensor> tensor13 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor14 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}}));
    std::shared_ptr<Tensor> tensor15 = (*tensor13) + tensor14;
    EXPECT_EQ(tensor15->shape(), std::vector<std::size_t>({2, 3}));
    EXPECT_EQ((*tensor15)(0, 0), 8.0);
    EXPECT_EQ((*tensor15)(0, 1), 10.0);
    EXPECT_EQ((*tensor15)(0, 2), 12.0);
    EXPECT_EQ((*tensor15)(1, 0), 14.0);
    EXPECT_EQ((*tensor15)(1, 1), 16.0);
    EXPECT_EQ((*tensor15)(1, 2), 18.0);
}

TEST(TensorTest, Matmul)
{
    // scalar x scalar
    std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(2.0);
    EXPECT_THROW((*tensor1) * tensor2, std::invalid_argument);

    // scalar x 1D
    std::shared_ptr<Tensor> tensor4 = std::make_shared<Tensor>(1.0);
    std::shared_ptr<Tensor> tensor5 = std::make_shared<Tensor>(std::vector<float>({2.0, 3.0, 4.0}));
    EXPECT_THROW((*tensor4) * tensor5, std::invalid_argument);

    // 1D * 1D
    std::shared_ptr<Tensor> tensor7 = std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 3.0}));
    std::shared_ptr<Tensor> tensor8 = std::make_shared<Tensor>(std::vector<float>({4.0, 5.0, 6.0}));
    std::shared_ptr<Tensor> tensor9 = (*tensor7) * tensor8;
    EXPECT_EQ(tensor9->shape(), std::vector<std::size_t>({}));
    EXPECT_EQ(tensor9->item(), 32.0);

    // 1D x 1D with mismatched dimensions
    std::shared_ptr<Tensor> tensor10 =
        std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 5.0}));
    std::shared_ptr<Tensor> tensor11 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}));
    EXPECT_THROW((*tensor10) * tensor11, std::invalid_argument);

    // 1D x 2D matching dimensions
    std::shared_ptr<Tensor> tensor12 = std::make_shared<Tensor>(std::vector<float>({1.0, 2.0}));
    std::shared_ptr<Tensor> tensor13 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}));
    std::shared_ptr<Tensor> tensor14 = (*tensor12) * tensor13;
    EXPECT_EQ(tensor14->shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*tensor14)(0), 18.0);
    EXPECT_EQ((*tensor14)(1), 21.0);
    EXPECT_EQ((*tensor14)(2), 24.0);

    // 2D x 1D matching dimensions
    std::shared_ptr<Tensor> tensor15 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor16 =
        std::make_shared<Tensor>(std::vector<float>({1.0, 2.0, 3.0}));
    std::shared_ptr<Tensor> tensor17 = (*tensor15) * tensor16;
    EXPECT_EQ(tensor17->shape(), std::vector<std::size_t>({2}));
    EXPECT_EQ((*tensor17)(0), 14.0);
    EXPECT_EQ((*tensor17)(1), 32.0);

    // 2D x 2D with mismatched dimensions
    std::shared_ptr<Tensor> tensor18 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor19 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    EXPECT_THROW((*tensor18) * tensor19, std::invalid_argument);

    // 2D x 2D with matching dimensions
    std::shared_ptr<Tensor> tensor20 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
    std::shared_ptr<Tensor> tensor21 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>({{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}));
    std::shared_ptr<Tensor> tensor22 = (*tensor20) * tensor21;
    EXPECT_EQ(tensor22->shape(), std::vector<std::size_t>({2, 2}));
    EXPECT_EQ((*tensor22)(0, 0), 58.0);
    EXPECT_EQ((*tensor22)(0, 1), 64.0);
    EXPECT_EQ((*tensor22)(1, 0), 139.0);
    EXPECT_EQ((*tensor22)(1, 1), 154.0);

    // 2D x 2D with large size
    std::shared_ptr<Tensor> tensor23 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>(200, std::vector<float>(300, 1.0)));
    std::shared_ptr<Tensor> tensor24 = std::make_shared<Tensor>(
        std::vector<std::vector<float>>(300, std::vector<float>(400, 1.0)));
    std::shared_ptr<Tensor> tensor25 = (*tensor23) * tensor24;
    EXPECT_EQ(tensor25->shape(), std::vector<std::size_t>({200, 400}));
    for (std::size_t i = 0; i < 200; i++)
    {
        for (std::size_t j = 0; j < 400; j++)
        {
            EXPECT_EQ((*tensor25)(i, j), 300.0);
        }
    }
}
TEST(TensorTest, BackwardAdditionAndMatmul)
{
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(2.0f, true);
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(3.0f, true);
    std::shared_ptr<Tensor> c = (*a) + b;
    c->backward();
    EXPECT_EQ(a->grad()[0], 1.0f);
    EXPECT_EQ(b->grad()[0], 1.0f);

    std::shared_ptr<Tensor> x =
        std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f}, true);
    std::shared_ptr<Tensor> y =
        std::make_shared<Tensor>(std::vector<float>{4.0f, 5.0f, 6.0f}, true);
    std::shared_ptr<Tensor> z = (*x) * y;
    z->backward();
    EXPECT_FLOAT_EQ(x->grad()[0], 4.0f);
    EXPECT_FLOAT_EQ(x->grad()[1], 5.0f);
    EXPECT_FLOAT_EQ(x->grad()[2], 6.0f);
    EXPECT_FLOAT_EQ(y->grad()[0], 1.0f);
    EXPECT_FLOAT_EQ(y->grad()[1], 2.0f);
    EXPECT_FLOAT_EQ(y->grad()[2], 3.0f);
}

TEST(ModuleTest, SerializationTest)
{
    class NeuralNetwork : public Module
    {
    private:
        // layers
        std::shared_ptr<Flatten> _flatten = std::make_shared<Flatten>();
        std::shared_ptr<Linear> _linear_1;
        std::shared_ptr<Linear> _linear_2;
        std::shared_ptr<Linear> _linear_3;
        // activation
        std::shared_ptr<Relu> _relu = std::make_shared<Relu>();

    public:
        NeuralNetwork(int seed)
        {
            _linear_1 = std::make_shared<Linear>(5 * 5, 5, seed);
            _linear_2 = std::make_shared<Linear>(5, 10, seed);
            _linear_3 = std::make_shared<Linear>(10, 10, seed);
            register_module("linear_1", _linear_1);
            register_module("linear_2", _linear_2);
            register_module("linear_3", _linear_3);
        }
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input)
        {
            std::shared_ptr<Tensor> flat = (*_flatten)(input);
            std::shared_ptr<Tensor> linear_1 = (*_linear_1)(flat);
            std::shared_ptr<Tensor> relu_1 = (*_relu)(linear_1);
            std::shared_ptr<Tensor> linear_2 = (*_linear_2)(relu_1);
            std::shared_ptr<Tensor> relu_2 = (*_relu)(linear_2);
            std::shared_ptr<Tensor> linear_3 = (*_linear_3)(relu_2);
            return linear_3;
        }
    };

    NeuralNetwork network_1(42);
    auto params = network_1.parameters();

    // Save state_dict
    auto state_dict = network_1.state_dict();
    save(state_dict, "state_dict.nn");

    NeuralNetwork network_2(77);
    auto params_2 = network_2.parameters();

    // confirm that params 1 is different from params 2 for layers 1, 3, 5
    EXPECT_NE(params[0].second->data()[0], params_2[0].second->data()[0]);
    EXPECT_NE(params[2].second->data()[0], params_2[2].second->data()[0]);
    EXPECT_NE(params[4].second->data()[0], params_2[4].second->data()[0]);

    // Load state_dict
    auto loaded_state_dict = load("state_dict.nn");
    network_2.load_state_dict(loaded_state_dict);

    // confirm that params 1 is the same as params 2 for layers 1, 3, 5
    EXPECT_EQ(params[0].second->data()[0], params_2[0].second->data()[0]);
    EXPECT_EQ(params[2].second->data()[0], params_2[2].second->data()[0]);
    EXPECT_EQ(params[4].second->data()[0], params_2[4].second->data()[0]);
    // delete state_dict file
    std::remove("state_dict.nn");
}

TEST(ModuleTest, Parameters)
{
    // Test simple module
    Linear linear(3, 2, 42); // in_features=3, out_features=2, seed=42
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2);
    EXPECT_EQ(params[0].first, "weight");
    EXPECT_EQ(params[1].first, "bias");

    // create a module with nested modules and parameters
    class NeuralNetwork : public Module
    {
    private:
        // layers
        std::shared_ptr<Flatten> _flatten = std::make_shared<Flatten>();
        std::shared_ptr<Linear> _linear_1 = std::make_shared<Linear>(28 * 28, 512);
        std::shared_ptr<Linear> _linear_2 = std::make_shared<Linear>(512, 512);
        std::shared_ptr<Linear> _linear_3 = std::make_shared<Linear>(512, 10);
        // activation
        std::shared_ptr<Relu> _relu = std::make_shared<Relu>();

    public:
        NeuralNetwork()
        {
            register_module("linear_1", _linear_1);
            register_module("linear_2", _linear_2);
            register_module("linear_3", _linear_3);
        }
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input)
        {
            std::shared_ptr<Tensor> flat = (*_flatten)(input);
            std::shared_ptr<Tensor> linear_1 = (*_linear_1)(flat);
            std::shared_ptr<Tensor> relu_1 = (*_relu)(linear_1);
            std::shared_ptr<Tensor> linear_2 = (*_linear_2)(relu_1);
            std::shared_ptr<Tensor> relu_2 = (*_relu)(linear_2);
            std::shared_ptr<Tensor> linear_3 = (*_linear_3)(relu_2);
            return linear_3;
        }
    };

    NeuralNetwork network;
    auto params_2 = network.parameters();
    EXPECT_EQ(params_2.size(), 6);
    EXPECT_EQ(params_2[0].first, "linear_1.weight");
    EXPECT_EQ(params_2[1].first, "linear_1.bias");
    EXPECT_EQ(params_2[2].first, "linear_2.weight");
    EXPECT_EQ(params_2[3].first, "linear_2.bias");
    EXPECT_EQ(params_2[4].first, "linear_3.weight");
    EXPECT_EQ(params_2[5].first, "linear_3.bias");
}

TEST(ModuleTest, Flatten)
{
    // flatten 0d tensor
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(2.0f);
    std::shared_ptr<Flatten> flatten = std::make_shared<Flatten>();
    std::shared_ptr<Tensor> b = (*flatten)(a);
    EXPECT_EQ(b->shape(), std::vector<std::size_t>({1}));
    EXPECT_EQ(b->item(), 2.0f);

    // flatten 1d tensor
    std::shared_ptr<Tensor> c = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
    std::shared_ptr<Tensor> d = (*flatten)(c);
    EXPECT_EQ(d->shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*d)(0), 1.0f);
    EXPECT_EQ((*d)(1), 2.0f);
    EXPECT_EQ((*d)(2), 3.0f);

    // flatten 2d tensor
    std::shared_ptr<Tensor> e =
        std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    std::shared_ptr<Tensor> f = (*flatten)(e);
    EXPECT_EQ(f->shape(), std::vector<std::size_t>({6}));
    EXPECT_EQ((*f)(0), 1.0f);
    EXPECT_EQ((*f)(1), 2.0f);
    EXPECT_EQ((*f)(2), 3.0f);
    EXPECT_EQ((*f)(3), 4.0f);
    EXPECT_EQ((*f)(4), 5.0f);
    EXPECT_EQ((*f)(5), 6.0f);
}

TEST(TensorTest, Linear)
{
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
    Linear linear(3, 2, 7);
    std::shared_ptr<Tensor> b = linear(a);
    EXPECT_EQ(b->shape(), std::vector<std::size_t>({2}));
    EXPECT_NEAR((*b)(0), -0.13753, 1e-5);
    EXPECT_NEAR((*b)(1), 2.26260, 1e-5);
}

TEST(ModuleTest, Relu)
{
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
    Relu relu;
    std::shared_ptr<Tensor> b = relu(a);
    EXPECT_EQ(b->shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*b)(0), 1.0f);
    EXPECT_EQ((*b)(1), 2.0f);
    EXPECT_EQ((*b)(2), 3.0f);

    std::shared_ptr<Tensor> c = std::make_shared<Tensor>(std::vector<float>{-1.0, -2.0, -3.0});
    std::shared_ptr<Tensor> d = relu(c);
    EXPECT_EQ(d->shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ((*d)(0), 0.0f);
    EXPECT_EQ((*d)(1), 0.0f);
    EXPECT_EQ((*d)(2), 0.0f);
}

TEST(ModuleTest, Softmax)
{
    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
    Softmax softmax;
    std::shared_ptr<Tensor> b = softmax(a);
    EXPECT_EQ(b->shape(), std::vector<std::size_t>({3}));
    EXPECT_NEAR((*b)(0), 0.09003, 1e-5);
    EXPECT_NEAR((*b)(1), 0.24473, 1e-5);
    EXPECT_NEAR((*b)(2), 0.66524, 1e-5);
}
TEST(LossTest, NLLLossForward)
{
    // Basic forward pass without grad
    std::shared_ptr<Tensor> input1 = std::make_shared<Tensor>(std::vector<float>{0.1f, 0.2f, 0.7f});
    NLLLoss nll_loss;
    std::size_t target1 = 2;
    std::shared_ptr<Tensor> loss1 = nll_loss(input1, target1);

    EXPECT_EQ(loss1->shape().size(), 0);
    EXPECT_FALSE(loss1->requires_grad());
    EXPECT_NEAR(loss1->item(), -std::log(0.7f), 1e-6);

    // Near-zero probability clamped
    std::shared_ptr<Tensor> input2 =
        std::make_shared<Tensor>(std::vector<float>{1e-15f, 0.5f, 0.5f});
    std::size_t target2 = 0;
    std::shared_ptr<Tensor> loss2 = nll_loss(input2, target2);
    EXPECT_NEAR(loss2->item(), -std::log(1e-12f), 1e-6);

    // Basic backward pass
    std::shared_ptr<Tensor> input_3 =
        std::make_shared<Tensor>(std::vector<float>{0.1f, 0.2f, 0.7f}, true);
    std::size_t target_3 = 2;
    std::shared_ptr<Tensor> loss_3 = nll_loss(input_3, target_3);

    EXPECT_TRUE(loss_3->requires_grad());
    loss_3->backward();

    ASSERT_EQ(input_3->grad().size(), 3);
    EXPECT_FLOAT_EQ(input_3->grad()[0], 0.0f);
    EXPECT_FLOAT_EQ(input_3->grad()[1], 0.0f);
    EXPECT_NEAR(input_3->grad()[2], -1.0f / 0.7f, 1e-6);

    // error handling
    std::shared_ptr<Tensor> input_4 =
        std::make_shared<Tensor>(std::vector<std::vector<float>>{{0.1f, 0.9f}});
    std::size_t target_4 = 0;
    EXPECT_THROW(nll_loss(input_4, target_4), std::runtime_error);

    // Test target out of bounds
    std::shared_ptr<Tensor> input_5 = std::make_shared<Tensor>(std::vector<float>{0.1f, 0.9f});
    std::size_t target_5 = 2;
    EXPECT_THROW(nll_loss(input_5, target_5), std::runtime_error);

    // Test target out of bounds (negative equivalent, size_t wrap around)
    std::shared_ptr<Tensor> input_6 = std::make_shared<Tensor>(std::vector<float>{0.1f, 0.9f});
    std::size_t target_6 = -1;
    EXPECT_THROW(nll_loss(input_6, target_6), std::runtime_error);
}

TEST(LossTest, CrossEntropyLossForward)
{
    // Basic forward pass without grad
    std::shared_ptr<Tensor> input1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
    CrossEntropyLoss ce_loss;
    std::size_t target1 = 2;
    std::shared_ptr<Tensor> loss1 = ce_loss(input1, target1);

    EXPECT_EQ(loss1->shape().size(), 0);
    EXPECT_FALSE(loss1->requires_grad());
    EXPECT_NEAR(loss1->item(), 0.40761f, 1e-5);

    // Basic backward pass
    std::shared_ptr<Tensor> input_2 =
        std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f}, true);
    std::size_t target_2 = 2;
    std::shared_ptr<Tensor> loss_2 = ce_loss(input_2, target_2);

    EXPECT_TRUE(loss_2->requires_grad());
    loss_2->backward();

    ASSERT_EQ(input_2->grad().size(), 3);
    EXPECT_NEAR(input_2->grad()[0], 0.09003f, 1e-5);
    EXPECT_NEAR(input_2->grad()[1], 0.24473f, 1e-5);
    EXPECT_NEAR(input_2->grad()[2], -0.33476f, 1e-5);

    // error handling: incorrect input shape
    std::shared_ptr<Tensor> input_3 =
        std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}});
    std::size_t target_3 = 0;
    EXPECT_THROW(ce_loss(input_3, target_3), std::runtime_error);

    // Test target out of bounds
    std::shared_ptr<Tensor> input_4 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f});
    std::size_t target_4 = 2;
    EXPECT_THROW(ce_loss(input_4, target_4), std::runtime_error);
}

TEST(SGDTest, Step)
{
    class NeuralNetwork : public Module
    {
    private:
        std::shared_ptr<Linear> _linear = std::make_shared<Linear>(5, 5, 7);
        std::shared_ptr<Relu> _relu = std::make_shared<Relu>();

    public:
        NeuralNetwork() { register_module("linear", _linear); }
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input)
        {
            std::shared_ptr<Tensor> linear = (*_linear)(input);
            std::shared_ptr<Tensor> relu = (*_relu)(linear);
            return relu;
        }
    };

    NeuralNetwork network;
    auto params = network.parameters();

    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>(5, 1.0f));

    // Copy linear weight
    std::vector<std::vector<float>> copy_linear_weight(
        params[0].second->shape()[0], std::vector<float>(params[0].second->shape()[1]));
    for (size_t i = 0; i < params[0].second->shape()[0]; i++)
    {
        for (size_t j = 0; j < params[0].second->shape()[1]; j++)
        {
            copy_linear_weight[i][j] = (*params[0].second)(i, j);
        }
    }

    // Copy linear bias
    std::vector<float> copy_linear_bias(params[1].second->shape()[0]);
    for (size_t i = 0; i < params[1].second->shape()[0]; i++)
    {
        copy_linear_bias[i] = (*params[1].second)(i);
    }

    auto copy_linear_weight_tensor = std::make_shared<Tensor>(copy_linear_weight);
    auto copy_linear_bias_tensor = std::make_shared<Tensor>(copy_linear_bias);

    auto xW = (*input) * copy_linear_weight_tensor;
    auto expected_linear_output = (*xW) + copy_linear_bias_tensor;

    SGD sgd(params);
    CrossEntropyLoss ce_loss;
    std::size_t target = 1;

    auto output = network(input);
    auto loss = ce_loss(output, target);
    loss->backward();

    if ((*expected_linear_output)(1) <= 0)
    {
        for (int i = 0; i < params[0].second->shape()[0]; i++)
        {
            float grad_val = (*params[0].second).grad()[i * params[0].second->stride()[0] + 1];
            EXPECT_EQ(grad_val, 0.0f) << "Expected gradient to be 0 at col 1, row " << i;
        }
    }

    sgd.step();

    // If gradient is 0, weight must be unchanged
    if ((*params[0].second).grad()[1] == 0)
    {
        float new_weight = (*params[0].second)(0, 1);
        EXPECT_FLOAT_EQ(new_weight, copy_linear_weight[0][1])
            << "Weight changed despite zero gradient";
    }

    // If gradient is > 0, weight should have decreased
    if ((*params[0].second).grad()[0] > 0)
    {
        float updated_weight = (*params[0].second)(0, 0);
        EXPECT_LT(updated_weight, copy_linear_weight[0][0])
            << "Weight not updated correctly for non-zero grad";
    }

    sgd.zero_grad();

    for (size_t i = 0; i < params[0].second->numel(); i++)
    {
        EXPECT_FLOAT_EQ((*params[0].second).grad()[i], 0.0f)
            << "Gradient not reset to 0 at index " << i;
    }
}