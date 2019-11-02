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

template<class T = int, size_t rank=0>
class Tensor {
public:

    Tensor()= default;

    explicit Tensor(std::vector<size_t> dims){
        std::cout << "hellooooooo";
        for(auto i : dims){
            std::cout << i;
        }
        if(rank == -1){
            _width = dims;
            auto temp = dims;
            temp[0] = 1;
            _strides = cummult(temp);
            _rank = dims.size();
        }
        else{
            dims.resize(rank);
            _width = dims;
            auto temp = dims;
            temp[0] = 1;
            _strides = cummult(temp);
            _rank = rank;
        }
    }

private:
    // to be shared
    // here things are hidden so you can decide how to access to them, don't be scared of being brave
    size_t _rank{};
    std::vector<T> _vec;
    std::vector<size_t> _strides;
    std::vector<size_t> _width;
};



#endif //TENSORLIB_TENSOR_H
