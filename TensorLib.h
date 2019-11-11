//
// Created by rr on 29/10/2019.
//

#ifndef TENSORLIB_TENSORLIB_H
#define TENSORLIB_TENSORLIB_H

#include <vector>
#include <algorithm>
#include <type_traits>
#include "utilities.h"
#include<cstdarg>


using namespace std;

template<class T = int, size_t rank=0>
class Tensor {
public:
    // when the rank is not specified
    Tensor<T>(std::initializer_list<size_t>&& a){
        // cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        if (rank!=0){
            assert(a.size()==rank);
        }
        widths = a;
        strides = cummult<size_t>(widths,1);
    }

    // when the rank is specified
    Tensor<T>(const std::vector<size_t>& a){
        // cout << "costruttore : Tensor<T>(std::vector<size_t>& a)" << endl;
        if (rank!=0){
            assert(a.size()==rank);
        }
        widths = a;
        strides = cummult<size_t>(widths,1);
    }

    // move constructor
    Tensor<T>(Tensor<T>&& a){
        // cout << "costruttore : Tensor<T>(Tensor<T>&& a)" << endl;
        if (rank!=0){
            assert(a._rank==rank);
        }
        widths = a.widths;
        strides = a.strides;
        data = a.data;
    }

    Tensor<T>(Tensor<T>& a){
        // cout << "costruttore : Tensor<T>(Tensor<T>& a)" << endl;
        if (rank!=0){
            assert(a._rank==rank);
        }
        widths = a.widths;
        strides = a.strides;
        data = a.data;
    }

    // initialize with an array that will be represented as a tensor
    void initialize(std::initializer_list<T>&& a){
        if ( (data.get() == nullptr) || (a.size() == data.get()->size()) ){
            data = make_shared<std::vector<T>>(a);
        }else{
            cout << "Una volta inizializzato il vettore non può essere modificato nelle dimensioni!" << endl;
        }
    }

    T operator()(initializer_list<size_t> a){
        
    }

    Tensor<T> slice(const size_t&  index, const size_t& value){
        Tensor<T> a = Tensor<T>(this);
        a.widths = erase<size_t>(widths, index);
        a.strides = erase<size_t>(strides, index);
        a.offset += (strides[index] * value);
        return a;
    }

    Tensor<T> flatten(const size_t& start, const size_t& stop){
        Tensor<T> a = Tensor<T>(this);
        a.widths = erase<size_t>(widths, index);
        a.strides = erase<size_t>(strides, index);
        a.offset += (strides[index] * value);
        return a;
    }
    
private:

    // metadata : immutable
    std::vector<size_t> widths;
    std::vector<size_t> strides;
    size_t offset=0;

    //data : mutable
    std::shared_ptr<std::vector<T>> data;
};





#endif //TENSORLIB_TENSORLIB_H
