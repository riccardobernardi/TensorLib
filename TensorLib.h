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
        cout << "costruttore : Tensor<T>(std::initializer_list<size_t>&& a)" << endl;
        if (rank!=0){
            assert(a.size()==rank);
        }
        _width = a;
        _strides = cummult<size_t>(_width,1);
        _rank = a.size();
        _size = cummult(_width)[0];
    }

    // when the rank is specified
    Tensor<T>(const std::vector<size_t>& a){
        cout << "costruttore : Tensor<T>(std::vector<size_t>& a)" << endl;
        if (rank!=0){
            assert(a.size()==rank);
        }
        _width = a;
        _strides = cummult<size_t>(_width,1);
        _rank = a.size();
        _size = cummult(_width)[0];
    }

    // move constructor
    Tensor<T>(Tensor<T>&& a){
        cout << "costruttore : Tensor<T>(Tensor<T>&& a)" << endl;
        if (rank!=0){
            assert(a._rank==rank);
        }
        _width = a._width;
        _strides = a._strides;
        _rank = a._rank;
        _size = a._size;
        _vec = a._vec;
    }

    // initialize with an array that will be represented as a tensor
    void initialize(std::initializer_list<T>&& a){
        assert(a.size() == _size);
        _vec = make_shared<std::vector<T>>(a);
    }

    void print_width(){
        for(auto i : _width){
            cout << i << endl;
        }
        cout << "*******" << endl;
    }

    void print_strides(){
        for(auto i : _strides){
            cout << i << endl;
        }
        cout << "*******" << endl;
    }

    void print_privates(){
        cout << "private " << "_rank " << _rank << endl;
        cout << "private " << "_size " << _size << endl;
        cout << "private " << "size del _default " << _default.size() << endl;
        for(auto i : _default){
            cout << "_default is " << get<0>(i) << " " << get<1>(i) << endl;
        }
    }

    size_t operator()(initializer_list<size_t> a){
        std::vector<size_t> b=a;
        for(auto i: _default){
            b.insert(b.begin() + get<0>(i), get<1>(i));
        }
        return ((*_vec)[sum(mult(_strides, b))]);
    }

    Tensor<T> slice(size_t index, size_t value){
        // have to check if there's another one index equal to mine, in this case i've to increase mine
        // why do i have to increase and not to decrease? Because im working in a subset and for example if i take out the second and again the second then the third(the second second) is the third!
        if (_default.size() != 0){
            cout << "Ho notato che hai fatto slicing a catena su un vettore, questa feature non è ancora testata!!";
            for(auto j : _default){
                for(auto i: _default){
                    if(index == get<0>(i)){
                        index++;
                    }
                }
            }
        }
        std::vector<size_t> new_width = erase(_width, index);
        Tensor<T> result = Tensor<T>(new_width);
        result._default.push_back(tuple<size_t, size_t>(index, value));
        result._strides = _strides;
        result._vec = _vec;
        return result;
    }

    Tensor<T> flatten(size_t start, size_t stop){
        std::vector<size_t> new_width;
        size_t tmp=1;
        for(int i=0;i<_width.size();++i){
            if(i>stop){
                new_width.push_back(tmp);
            }
            if (i<start || i>stop){
                new_width.push_back(_width[i]);
            }else{
                tmp*=_width[i];
            }
        }
        Tensor<T> result = Tensor<T>(new_width);
        if(result._old_dimensions == nullptr){
            //because if i repeat flattening 2 times on the same vector i dont need to replace many times the dimensions, the real vector is only the really first one
            result._old_dimensions = _width;
        }
        return result;
    }

private:

    size_t _rank=0;
    size_t _size=0;

    // when you do slicing then there is an index that it is defaulted
    std::vector<tuple<size_t, size_t>> _default;
    std::vector<size_t> _old_dimensions;

    std::shared_ptr<std::vector<T>> _vec;
    std::vector<size_t> _strides;
    std::vector<size_t> _width;
};





#endif //TENSORLIB_TENSORLIB_H
