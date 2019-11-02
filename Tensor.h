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

class Tensor {
public:
    template<typename T> Tensor(std::vector<size_t> dims){
        _width = dims;
        auto temp = dims;
        temp[0] = 1;
        _strides = cummult(dims);
        _rank = dims.size();
    }

    template<typename T, size_t rank=0> Tensor(std::array<size_t,rank> dims){
        dims = slice(dims,0, rank);
        _width = dims;
        auto temp = dims;
        temp[0] = 1;
        _strides = cummult(dims);
        _rank = rank;
    }
private:
    // to be shared
    size_t _rank;
    std::vector<int> _vec;
    std::vector<size_t> _strides;
    std::vector<size_t> _width;

};



#endif //TENSORLIB_TENSOR_H
