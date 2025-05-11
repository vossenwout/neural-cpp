#include "nn/tensor.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

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