//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H

template <typename T>
template <typename T, int rank>
class Tensor {
private:
    // to be shared
    std::Array<T> _vec;
    std::Array<T> _strides;
    std::Array<T> _width;
public:
    //methods
    Tensor(std::Array<int> dims){
        _width = dims;
    }

    Tensor(std::Array<int> dims){
        _width = dims;
    }
};



#endif //TENSORLIB_TENSOR_H
