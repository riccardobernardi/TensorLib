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

    // copy constructor
    Tensor<T>(const Tensor<T>& a){
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
        assert(a.size() == widths.size());

        std::vector<size_t> b = a;
        size_t tmp = 0;
        for(size_t i=0; i< b.size(); ++i){
            assert(b[i] < widths[i] && b[i] >= 0);
            tmp += b[i] * strides[i];
        }

        tmp += offset;

        return (data.get()->at(tmp));
    }

    Tensor<T> slice(const size_t&  index, const size_t& value){
        Tensor<T> a = Tensor<T>(widths);
        a.widths = erase<size_t>(widths, index);
        a.strides = erase<size_t>(strides, index);
        a.offset += (strides[index] * value);
        return a;
    }

    Tensor<T> flatten(const size_t& start, const size_t& stop){
        std::vector<size_t> new_width;
        size_t tmp=1;
        size_t flat=-1;

        for(int i=0;i<widths.size();++i){
            if (i<start || i>stop){
                new_width.push_back(widths[i]);
            }else{
                tmp*=widths[i];
                if(i==stop){
                    new_width.push_back(tmp);
                    // flat = new_width.size() - 1;
                }
            }
        }

        Tensor<T> a = Tensor<T>(new_width);

        // TODO opt
        a.strides = cummult(new_width);
        a.data = data;

        return a;
    }

    Tensor<T> window(const size_t& index, const size_t& start, const size_t& stop){
        assert(stop > start);
        assert(widths[index] > stop);
        assert(start >= 0);

        Tensor<T> a = Tensor<T>(widths);

        a.widths[index] = stop - start + 1;
        a.offset += a.strides[index] * start;

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
