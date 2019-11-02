//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H

#include <vector>

template<class T>
static std::vector<T> cummult(std::vector<T> a){
    for(int i=1; i< a.size(); ++i){
        a[i] = a[i-1] * a[i];
    }

    return a;
}

template<typename T>
class Tensor {
public:
    Tensor(std::vector<size_t> dims){
        _width = dims;
        auto temp = dims;
        temp[0] = 1;
        _strides = cummult(dims);
        _rank = dims.size();
    }
private:
    // to be shared
    size_t _rank;
    std::shared_ptr<std::vector<T>> _vec;
    std::shared_ptr<std::vector<size_t>> _strides;
    std::shared_ptr<std::vector<size_t>> _width;
};

#endif //TENSORLIB_TENSOR_H
