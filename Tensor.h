//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H

#include <vector>

using namespace std;

template<class T>
static std::vector<T> cummult(std::vector<T> a){
    for(int i=1; i< a.size(); ++i){
        a[i] = a[i-1] * a[i];
    }

    return a;
}

template<class T = int, size_t rank=0>
class Tensor {
public:
    // when the rank is not specified
    Tensor(std::initializer_list<size_t>&& a){
        _width = a;
        _strides = cummult(_width);
        _rank = a.size();
        cout << to_string(_rank);
    }

    // when the rank is specified
    Tensor(int a...){
        _width = a;
        _strides = cummult(_width);
        _rank = a;
        cout << to_string(_rank);
    }

private:
    size_t _rank=0;
    std::shared_ptr<std::vector<T>> _vec;
    std::vector<size_t> _strides;
    std::vector<size_t> _width;
};



#endif //TENSORLIB_TENSOR_H
