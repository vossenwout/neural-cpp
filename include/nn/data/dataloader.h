#pragma once
#include "nn/data/datasets.h"
#include <memory>
#include <utility>
#include <vector>

class DataLoader
{
private:
    Dataset *_dataset;
    int _batch_size;
    std::vector<int> _indices;

public:
    DataLoader(Dataset *dataset, int batch_size, bool shuffle = true);

    class Iterator
    {
    private:
        DataLoader *_dataloader;
        int _index;

    public:
        Iterator(DataLoader *dataloader, int index);
        void operator++();
        std::vector<std::pair<int, std::shared_ptr<Tensor>>> operator*();
        bool operator!=(const Iterator &other);
    };

    Iterator begin();
    Iterator end();

    std::size_t batch_size() const;
    std::size_t n_samples() const;
    std::size_t n_batches() const;
};