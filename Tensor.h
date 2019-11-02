//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H

#include <vector>

template <typename T, size_t rank=0>
class Tensor {
private:
    // to be shared
    std::shared_ptr<std::vector<T>> _vec;
    std::shared_ptr<std::vector<size_t>> _strides;
    std::shared_ptr<std::vector<size_t>> _width;
public:
    Tensor(std::vector<size_t> dims){
        _width = dims;
    }
};

#endif //TENSORLIB_TENSOR_H
