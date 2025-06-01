#include "nn/data/dataloader.h"
#include "nn/data/datasets.h"
#include <algorithm>
#include <random>

DataLoader::DataLoader(Dataset *dataset, int batch_size, bool shuffle)
    : _dataset(dataset), _batch_size(batch_size)
{
    _indices.resize(_dataset->get_length());
    std::iota(_indices.begin(), _indices.end(), 0);
    if (shuffle)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(_indices.begin(), _indices.end(), g);
    }
}

DataLoader::Iterator::Iterator(DataLoader *dataloader, int index)
    : _dataloader(dataloader), _index(index)
{
}

void DataLoader::Iterator::operator++() { _index += _dataloader->_batch_size; }

std::vector<std::pair<int, std::shared_ptr<Tensor>>> DataLoader::Iterator::operator*()
{
    std::vector<std::pair<int, std::shared_ptr<Tensor>>> batch;
    for (int i = 0; i < _dataloader->_batch_size; i++)
    {
        batch.push_back(_dataloader->_dataset->get_item(_dataloader->_indices[_index + i]));
    }
    return batch;
}

bool DataLoader::Iterator::operator!=(const Iterator &other) { return _index != other._index; }

DataLoader::Iterator DataLoader::begin() { return Iterator(this, 0); }

DataLoader::Iterator DataLoader::end() { return Iterator(this, _dataset->get_length()); }

std::size_t DataLoader::batch_size() const { return _batch_size; }

std::size_t DataLoader::n_samples() const { return _dataset->get_length(); }

std::size_t DataLoader::n_batches() const
{
    return (n_samples() + batch_size() - 1) / batch_size();
}